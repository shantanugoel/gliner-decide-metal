// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "gliner-decide-metal",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "GLiNERDecideCore", targets: ["GLiNERDecideCore"]),
        .executable(name: "gliner-decide-metal", targets: ["gliner-decide-metal"]),
        .executable(name: "gliner-decide-raw", targets: ["gliner-decide-raw"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            revision: "901941965d82e4a216d4d117231d847d194c563d"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            exact: "1.3.4"
        )
    ],
    targets: [
        .target(
            name: "GLiNERDecideCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        ),
        .executableTarget(
            name: "gliner-decide-metal",
            dependencies: [
                "GLiNERDecideCore",
                .product(name: "MLX", package: "mlx-swift")
            ]
        ),
        .executableTarget(
            name: "gliner-decide-raw",
            dependencies: [
                "GLiNERDecideCore",
                .product(name: "MLX", package: "mlx-swift")
            ]
        ),
        .testTarget(
            name: "GLiNERDecideCoreTests",
            dependencies: ["GLiNERDecideCore"]
        )
    ]
)
