import Foundation
import MLX
import MLXFast

/// Optional custom kernels used by the benchmark. The default path remains
/// MLX's optimized LayerNorm so correctness can be compared independently.
public final class FusedKernels: @unchecked Sendable {
    private let residualLayerNormKernel: MLXFast.MLXFastKernel
    private let relativeBiasKernel: MLXFast.MLXFastKernel

    public init() {
        relativeBiasKernel = MLXFast.metalKernel(
            name: "gliner_decide_relative_bias",
            inputNames: [
                "q", "k", "position_key", "position_query",
                "c2p_indices", "p2c_indices", "key_mask", "scale",
            ],
            outputNames: ["out"],
            source: """
                uint global = thread_position_in_grid.x;
                uint lane = global % 32;
                uint row = global / 32;
                uint b = row / (H * L);
                uint h = (row / L) % H;
                uint query = row % L;
                float q_cache[C];
                uint q_base = ((b * H + h) * L + query) * C;
                for (uint d = 0; d < C; ++d) {
                    q_cache[d] = float(q[q_base + d]);
                }

                for (uint key_index = lane; key_index < L; key_index += 32) {
                    uint k_base = ((b * H + h) * L + key_index) * C;
                    uint c2p_index = c2p_indices[(h * L + query) * L + key_index];
                    uint p2c_index = p2c_indices[(h * L + key_index) * L + query];
                    uint pos_k_base = (h * P + c2p_index) * C;
                    uint pos_q_base = (h * P + p2c_index) * C;
                    float c2p = 0.0f;
                    float p2c = 0.0f;
                    for (uint d = 0; d < C; ++d) {
                        float kd = float(k[k_base + d]);
                        c2p += q_cache[d] * float(position_key[pos_k_base + d]);
                        p2c += kd * float(position_query[pos_q_base + d]);
                    }
                    float score = (c2p + p2c) * scale[0];
                    if (key_mask[b * L + key_index] < -60000.0f) {
                        score = -65504.0f;
                    }
                    out[((b * H + h) * L + query) * L + key_index] = T(score);
                }
                """
        )

        residualLayerNormKernel = MLXFast.metalKernel(
            name: "gliner_decide_residual_layer_norm",
            inputNames: ["x", "residual", "weight", "bias", "eps"],
            outputNames: ["out"],
            source: """
                uint global = thread_position_in_grid.x;
                uint row = global / 32;
                uint lane = global % 32;
                uint base = row * C;
                float total = 0.0f;

                for (uint j = lane; j < C; j += 32) {
                    total += float(x[base + j]) + float(residual[base + j]);
                }
                total = simd_sum(total);
                float mean = total / float(C);

                float variance = 0.0f;
                for (uint j = lane; j < C; j += 32) {
                    float delta = float(x[base + j]) + float(residual[base + j]) - mean;
                    variance += delta * delta;
                }
                variance = simd_sum(variance) / float(C);

                float inverse = rsqrt(variance + eps[0]);
                for (uint j = lane; j < C; j += 32) {
                    float value = float(x[base + j]) + float(residual[base + j]);
                    out[base + j] = T((value - mean) * inverse * float(weight[j]) + float(bias[j]));
                }
                """
        )
    }

    /// Computes DeBERTa's content-to-position plus position-to-content bias.
    /// The content QK product is intentionally left to MLX's fused SDPA.
    public func relativeBias(
        q: MLXArray,
        k: MLXArray,
        positionKey: MLXArray,
        positionQuery: MLXArray,
        c2pIndices: MLXArray,
        p2cIndices: MLXArray,
        keyMask: MLXArray,
        scale: Float,
        batch: Int,
        heads: Int,
        sequenceLength: Int,
        headDimension: Int,
        positionCount: Int
    ) -> MLXArray {
        precondition(q.shape == k.shape)
        precondition(q.shape.count == 4)
        precondition(q.shape[2] == sequenceLength)
        precondition(q.shape[3] == headDimension)
        let scaleArray = MLXArray([Float(scale)]).asType(.float32)
        return relativeBiasKernel(
            [
                q,
                k,
                positionKey,
                positionQuery,
                c2pIndices,
                p2cIndices,
                keyMask,
                scaleArray,
            ],
            template: [
                ("B", batch),
                ("H", heads),
                ("L", sequenceLength),
                ("C", headDimension),
                ("P", positionCount),
                ("T", q.dtype),
            ],
            grid: (batch * heads * sequenceLength * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[batch, heads, sequenceLength, sequenceLength]],
            outputDTypes: [q.dtype]
        )[0]
    }

    /// Computes `LayerNorm(x + residual, weight, bias)` in one Metal kernel.
    public func residualLayerNorm(
        _ x: MLXArray,
        _ residual: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        eps: Float
    ) -> MLXArray {
        let shape = x.shape
        precondition(shape == residual.shape, "residual shape mismatch")
        precondition(shape.count >= 1, "x must have at least one dimension")
        let columns = shape[shape.count - 1]
        precondition(columns > 0 && columns % 32 == 0, "hidden size must be a positive multiple of 32")
        let rows = shape.dropLast().reduce(1, *)
        let epsArray = MLXArray([Float(eps)]).asType(.float32)
        return residualLayerNormKernel(
            [x, residual, weight, bias, epsArray],
            template: [("C", columns), ("T", x.dtype)],
            // MLX's grid is total threads (not threadgroups). Each row uses
            // one 32-wide simdgroup, so dispatch 32 threads per row.
            grid: (rows * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [shape],
            outputDTypes: [x.dtype],
            verbose: false
        )[0]
    }
}
