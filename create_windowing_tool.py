import numpy as np
import argparse
import os
import json

# 定义用于mu->HU转换的参考点
HU_AIR = -1000.0
HU_HIGH_DENSITY = 1000.0

# --- HTML, CSS, 和 JavaScript 模板 ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Windowing Tool (v2)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #f0f2f5;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 800px;
        }}
        h1, h2 {{
            text-align: center;
            color: #333;
        }}
        .canvas-container {{
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
            background: #222;
            padding: 10px;
            border-radius: 4px;
        }}
        canvas {{
            border: 1px solid #ccc;
            image-rendering: pixelated; /* 保持像素清晰 */
            width: 100%;
            max-width: {width}px;
        }}
        .controls {{
            display: grid;
            grid-template-columns: 100px 1fr 80px;
            gap: 10px;
            align-items: center;
            margin-bottom: 15px;
        }}
        label {{
            font-weight: bold;
            color: #555;
        }}
        input[type="range"] {{
            width: 100%;
            cursor: pointer;
        }}
        .value-display {{
            font-family: "SF Mono", "Courier New", monospace;
            background-color: #e9ecef;
            padding: 5px 10px;
            border-radius: 4px;
            text-align: center;
        }}
        .info {{
            border-top: 1px solid #eee;
            padding-top: 15px;
            margin-top: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Windowing Tool</h1>
        <div class="canvas-container">
            <canvas id="projectionCanvas" width="{width}" height="{height}"></canvas>
        </div>

        <div class="controls">
            <label for="wlSlider">Window Level:</label>
            <input type="range" id="wlSlider" min="{min_hu}" max="{max_hu}" value="{default_wl}" step="1">
            <span id="wlValue" class="value-display">{default_wl}</span>
        </div>

        <div class="controls">
            <label for="wwSlider">Window Width:</label>
            <input type="range" id="wwSlider" min="1" max="{ww_max_value}" value="{default_ww}" step="1">
            <span id="wwValue" class="value-display">{default_ww}</span>
        </div>

        <div class="info">
            <p><strong>File:</strong> {filename}</p>
            <p><strong>Dimensions:</strong> {width} x {height}</p>
            <p><strong>HU Range:</strong> [{min_hu}, {max_hu}]</p>
        </div>
    </div>

    <script>
        // --- 数据注入区 ---
        const huData = {hu_data_json};
        const canvasWidth = {width};
        const canvasHeight = {height};
        // --- 数据注入区结束 ---

        const canvas = document.getElementById('projectionCanvas');
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvasWidth, canvasHeight);

        const wlSlider = document.getElementById('wlSlider');
        const wwSlider = document.getElementById('wwSlider');
        const wlValueSpan = document.getElementById('wlValue');
        const wwValueSpan = document.getElementById('wwValue');

        function drawImage() {{
            const wl = parseFloat(wlSlider.value);
            const ww = parseFloat(wwSlider.value);

            const lowerBound = wl - ww / 2;
            const upperBound = wl + ww / 2;

            for (let i = 0; i < huData.length; i++) {{
                const huValue = huData[i];
                let intensity = 0;

                if (huValue > lowerBound) {{
                    intensity = (huValue - lowerBound) / ww;
                }}

                // Clip the value to [0, 1] and scale to [0, 255]
                intensity = Math.max(0, Math.min(1, intensity));
                const pixelValue = Math.round(intensity * 255);

                const pixelIndex = i * 4;
                imageData.data[pixelIndex] = pixelValue;     // R
                imageData.data[pixelIndex + 1] = pixelValue; // G
                imageData.data[pixelIndex + 2] = pixelValue; // B
                imageData.data[pixelIndex + 3] = 255;        // A
            }}
            ctx.putImageData(imageData, 0, 0);
        }}

        function updateValuesAndDraw() {{
            wlValueSpan.textContent = wlSlider.value;
            wwValueSpan.textContent = wwSlider.value;
            drawImage();
        }}

        wlSlider.addEventListener('input', updateValuesAndDraw);
        wwSlider.addEventListener('input', updateValuesAndDraw);

        // Initial draw on page load
        window.onload = updateValuesAndDraw;
    </script>
</body>
</html>
"""


def create_tool(npy_path, output_html_path):
    """
    加载NPY文件，转换为HU值，并生成一个交互式的HTML工具。
    """
    print(f"--- Loading NPY file: {npy_path} ---")
    if not os.path.exists(npy_path):
        print(f"Error: File not found at {npy_path}")
        return

    # 1. 加载并转换为HU值
    proj_data = np.load(npy_path)
    proj_data = np.nan_to_num(proj_data)

    mu_min, mu_max = np.min(proj_data), np.max(proj_data)
    if mu_max - mu_min < 1e-6:
        a, b = 0, HU_AIR
    else:
        a = (HU_HIGH_DENSITY - HU_AIR) / (mu_max - mu_min)
        b = HU_AIR - a * mu_min
    hu_image = a * proj_data + b

    height, width = hu_image.shape
    min_hu, max_hu = int(np.floor(np.min(hu_image))), int(np.ceil(np.max(hu_image)))

    # --- 【关键修正】 ---
    # 旧方法: ww_range = max_hu - min_hu
    # 新方法: 使用一个固定的、足够大的值，确保能设置宽骨窗
    WW_SLIDER_MAX = 5000
    # --- 【修正结束】 ---

    # 2. 设置合理的默认值
    default_wl = int((min_hu + max_hu) / 2)
    default_ww = int((max_hu - min_hu) * 0.4)  # 默认窗宽仍基于数据范围，但滑块上限更高

    # 3. 将2D HU数据扁平化并转换为JSON列表
    hu_data_flat = hu_image.flatten().tolist()
    hu_data_json = json.dumps(hu_data_flat)

    print(f"Image Dimensions: {width}x{height}")
    print(f"Calculated HU Range: [{min_hu}, {max_hu}]")
    print(f"Window Width Slider Max set to: {WW_SLIDER_MAX}")

    # 4. 填充HTML模板
    final_html = HTML_TEMPLATE.format(
        width=width,
        height=height,
        min_hu=min_hu,
        max_hu=max_hu,
        ww_max_value=WW_SLIDER_MAX,  # <--- 使用新的固定最大值
        default_wl=default_wl,
        default_ww=default_ww,
        filename=os.path.basename(npy_path),
        hu_data_json=hu_data_json
    )

    # 5. 写入文件
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"\n--- Success! ---")
    print(f"Interactive tool has been re-generated at: {output_html_path}")
    print("You can now open this HTML file in your web browser.")


def main():
    parser = argparse.ArgumentParser(
        description="Create an interactive HTML tool to find the optimal Window Width (WW) and Window Level (WL) for a .npy projection file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input_npy',
        type=str,
        help="Path to the input .npy projection file."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='windowing_tool1.html',
        help="Path for the output HTML file. (Default: windowing_tool.html)"
    )
    args = parser.parse_args()
    create_tool(args.input_npy, args.output)


if __name__ == '__main__':
    main()