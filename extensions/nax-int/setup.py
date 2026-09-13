from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    setup(
        name="mlx-nax-int",
        version="0.0.0",
        description="Experimental native integer NAX matmul for MLX.",
        ext_modules=[extension.CMakeExtension("mlx_nax_int._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mlx_nax_int"],
        package_data={"mlx_nax_int": ["*.so", "*.dylib", "*.metallib"]},
        # Kept in lockstep with the build-system bound in pyproject.toml; see the
        # comment there for why this tracks tested versions rather than a range.
        install_requires=["mlx==0.32.0"],
        zip_safe=False,
        python_requires=">=3.10",
    )
