import numpy as np
import tigre
import tigre.algorithms as algs
import time

# 这个脚本现在是完全自包含的，定义了所有必需的参数。

print("--- TIGRE Sanity Check (v4) ---")
print("Running a self-contained TIGRE example with a generated phantom and explicit angles...")

try:
    # --- 1. 创建几何 ---
    # 使用TIGRE的默认几何。
    geo = tigre.geometry_default(high_resolution=False)

    # --- 1.5. 定义扫描角度 (这是关键的新增部分) ---
    # 创建一个从0到2*pi（360度）的100个等间距角度数组。
    # 这是之前脚本中缺失的部分。
    angles = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)
    # 将角度数组赋值给几何对象
    geo.angles = angles

    print("\n--- TIGRE Default Geometry ---")
    print(geo)
    print(f"\n--- Scan Angles ---")
    print(f"Defined {len(geo.angles)} angles from {geo.angles.min():.2f} to {geo.angles.max():.2f} radians.")
    print("----------------------------\n")

    # --- 2. 创建一个简单的测试模型 (Phantom) ---
    # 在内存中创建一个中心有立方体的模型。
    print("Generating a simple cube phantom in memory...")
    phantom = np.zeros(geo.nVoxel, dtype=np.float32)
    cube_size = np.array(geo.nVoxel) // 4
    center = np.array(geo.nVoxel) // 2
    start_indices = center - cube_size // 2
    end_indices = center + cube_size // 2
    phantom[start_indices[0]:end_indices[0],
    start_indices[1]:end_indices[1],
    start_indices[2]:end_indices[2]] = 1.0
    print(f"Phantom created with shape {phantom.shape} and a central cube.\n")

    # --- 3. 前向投影 ---
    # 现在 geo.angles 是一个有效的数组，这个调用应该可以进入CUDA代码了。
    print("Generating projection data using TIGRE's forward projection (Ax)...")
    projections = tigre.Ax(phantom, geo, geo.angles)
    print("Projection data generated successfully.\n")

    # --- 4. 运行重建 ---
    niter = 20
    print(f"Running SART reconstruction for {niter} iterations...")
    t0 = time.time()
    imgSART = algs.sart(projections, geo, geo.angles, niter)
    t1 = time.time()

    print(f"\n\n--- TIGRE Sanity Check Successful! ---")
    print(f"Reconstruction completed in {t1 - t0:.2f} seconds.")
    print(f"Reconstructed image shape: {imgSART.shape}")
    print(f"Mean value of reconstructed image: {np.mean(imgSART)}")
    print("If you see this message, your TIGRE installation and GPU environment are working correctly.")
    print("--------------------------------------\n")

except Exception as e:
    print("\n\n--- TIGRE Sanity Check FAILED! ---")
    print("An error occurred while running a standard TIGRE example.")
    print("This strongly indicates an issue with your TIGRE/CUDA/Driver installation.")
    print("The error happened during the core Ax or SART operations.")
    print("\nError details:")
    import traceback

    traceback.print_exc()
    print("------------------------------------\n")