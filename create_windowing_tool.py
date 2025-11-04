import numpy as np
import argparse
import json
import os
import sys

# 升级后的HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>窗宽窗位调节器</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #282c34;
            color: #abb2bf;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            box-sizing: border-box;
        }}
        .container {{
            width: 100%;
            max-width: 900px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1, h2 {{
            color: #61afef;
            text-align: center;
            word-break: break-all;
        }}
        h2 {{
            font-size: 1rem;
            font-weight: normal;
            color: #98c379;
        }}
        .controls {{
            background-color: #3a3f4b;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            width: 100%;
            max-width: 512px;
            box-sizing: border-box;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .control-group {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }}
        .control-group label {{
            min-width: 150px; /* 增加宽度以容纳更长的标签 */
            font-weight: bold;
        }}
        input[type="range"] {{
            flex-grow: 1;
            margin: 0 15px;
            cursor: pointer;
        }}
        .value-display {{
            min-width: 80px;
            font-family: 'Courier New', Courier, monospace;
            background-color: #282c34;
            padding: 5px;
            border-radius: 4px;
            text-align: center;
        }}
        canvas {{
            background-color: #000;
            border: 1px solid #444;
            border-radius: 4px;
            margin-top: 20px;
            max-width: 100%;
            height: auto;
            image-rendering: pixelated;
            cursor: grab; /* 提示用户可以拖动 */
        }}
        canvas:active {{
            cursor: grabbing; /* 拖动时的光标样式 */
        }}
        .stats, .instructions {{
            background-color: #3a3f4b;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            width: 100%;
            max-width: 512px;
            box-sizing: border-box;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.9rem;
        }}
        .stats p, .instructions p {{
            margin: 5px 0;
        }}
        .instructions {{
            background-color: #4b5263;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>窗宽窗位调节器</h1>
        <h2>{file_path}</h2>

        <div class="instructions">
            <p><strong>操作提示:</strong></p>
            <p>1. 使用下方滑块进行精确调节。</p>
            <p>2. <strong>在图像上按住并拖动鼠标:</strong></p>
            <p>&nbsp;&nbsp;- 左右拖动: 调节 <strong>窗位</strong></p>
            <p>&nbsp;&nbsp;- 上下拖动: 调节 <strong>窗宽</strong></p>
        </div>

        <canvas id="canvas"></canvas>

        <div class="controls">
            <div class="control-group">
                <label for="levelSlider">窗位 (Window Level):</label>
                <input type="range" id="levelSlider" min="{slider_min_level}" max="{slider_max_level}" step="{slider_step}" value="{default_level}">
                <span id="levelValue" class="value-display">{default_level}</span>
            </div>
            <div class="control-group">
                <label for="widthSlider">窗宽 (Window Width):</label>
                <input type="range" id="widthSlider" min="{slider_min_width}" max="{slider_max_width}" step="{slider_step}" value="{default_width}">
                <span id="widthValue" class="value-display">{default_width}</span>
            </div>
        </div>

        <div class="stats">
            <p><strong>数据统计:</strong></p>
            <p>尺寸 (Shape): {data_shape}</p>
            <p>最小值 (Min): {data_min:.6f}</p>
            <p>最大值 (Max): {data_max:.6f}</p>
            <p>平均值 (Mean): {data_mean:.6f}</p>
            <p>标准差 (Std): {data_std:.6f}</p>
        </div>
    </div>

    <script>
        // --- 数据和常量 ---
        const rawData = JSON.parse('{json_data}');
        const imageWidth = {img_width};
        const imageHeight = {img_height};

        // --- DOM 元素 ---
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const levelSlider = document.getElementById('levelSlider');
        const widthSlider = document.getElementById('widthSlider');
        const levelValueSpan = document.getElementById('levelValue');
        const widthValueSpan = document.getElementById('widthValue');

        // --- 初始化 Canvas ---
        canvas.width = imageWidth;
        canvas.height = imageHeight;
        const imageData = ctx.createImageData(imageWidth, imageHeight);
        const imageDataBuffer = imageData.data;

        // --- 核心绘图函数 ---
        function updateImage() {{
            const level = parseFloat(levelSlider.value);
            const width = parseFloat(widthSlider.value);

            levelValueSpan.textContent = level.toFixed(4);
            widthValueSpan.textContent = width.toFixed(4);

            const windowMin = level - (width / 2);
            const windowMax = level + (width / 2);

            for (let i = 0; i < rawData.length; i++) {{
                const row = rawData[i];
                for (let j = 0; j < row.length; j++) {{
                    const pixelValue = row[j];
                    const dataIndex = (i * imageWidth + j) * 4;

                    let grayValue = 0;

                    if (pixelValue <= windowMin) {{
                        grayValue = 0;
                    }} else if (pixelValue >= windowMax) {{
                        grayValue = 255;
                    }} else {{
                        if (width > 0) {{
                            grayValue = ((pixelValue - windowMin) / width) * 255;
                        }} else {{
                            grayValue = 255; // 如果窗宽为0，则显示为白色
                        }}
                    }}

                    grayValue = Math.max(0, Math.min(255, grayValue));

                    imageDataBuffer[dataIndex] = grayValue;
                    imageDataBuffer[dataIndex + 1] = grayValue;
                    imageDataBuffer[dataIndex + 2] = grayValue;
                    imageDataBuffer[dataIndex + 3] = 255;
                }}
            }}

            ctx.putImageData(imageData, 0, 0);
        }}

        // --- 事件监听 ---
        levelSlider.addEventListener('input', updateImage);
        widthSlider.addEventListener('input', updateImage);

        // --- 新增：鼠标拖拽调节功能 ---
        let isDragging = false;
        let startX, startY;
        let startLevel, startWidth;

        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startLevel = parseFloat(levelSlider.value);
            startWidth = parseFloat(widthSlider.value);
            // 阻止默认的拖拽行为，例如拖拽图片
            e.preventDefault();
        }});

        document.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;

            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;

            // 灵敏度调整：可以修改这里的乘数来改变拖拽的灵敏度
            // 左右拖动改变窗位 (Level)
            const levelChange = deltaX * (parseFloat(levelSlider.max) - parseFloat(levelSlider.min)) / 800;
            let newLevel = startLevel + levelChange;

            // 上下拖动改变窗宽 (Width)，注意Y轴方向
            const widthChange = -deltaY * (parseFloat(widthSlider.max) - parseFloat(widthSlider.min)) / 800;
            let newWidth = startWidth + widthChange;

            // 限制新值在滑块的范围内
            newLevel = Math.max(parseFloat(levelSlider.min), Math.min(parseFloat(levelSlider.max), newLevel));
            newWidth = Math.max(parseFloat(widthSlider.min), Math.min(parseFloat(widthSlider.max), newWidth));

            // 更新滑块的值，这会自动触发 'input' 事件并调用 updateImage
            levelSlider.value = newLevel;
            widthSlider.value = newWidth;

            // 手动触发input事件以确保UI同步更新
            levelSlider.dispatchEvent(new Event('input'));
            widthSlider.dispatchEvent(new Event('input'));
        }});

        document.addEventListener('mouseup', () => {{
            isDragging = false;
        }});

        // --- 初始加载 ---
        window.onload = updateImage;
    </script>
</body>
</html>
"""


def create_viewer(npy_path, output_path):
    print(f"--- 正在分析文件: {npy_path} ---")
    try:
        data = np.load(npy_path)
    except FileNotFoundError:
        print(f"错误: 文件未找到 '{npy_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载NPY文件失败。 {e}")
        sys.exit(1)

    if data.ndim != 2:
        print(f"错误: 此脚本仅支持2D NumPy数组，但您的数据维度为 {data.ndim}。")
        sys.exit(1)

    data_min = float(data.min())
    data_max = float(data.max())
    data_mean = float(data.mean())
    data_std = float(data.std())
    img_height, img_width = data.shape

    print("\n--- 数据统计 ---")
    print(f"数据类型 (dtype): {data.dtype}")
    print(f"数据形状 (shape): {data.shape}")
    print(f"最小值: {data_min:.6f}")
    print(f"最大值: {data_max:.6f}")
    print(f"平均值: {data_mean:.6f}")
    print(f"标准差: {data_std:.6f}")

    data_range = data_max - data_min

    slider_min_level = data_min
    slider_max_level = data_max

    # 窗宽的最小值不能为0，设为一个很小的正数
    slider_min_width = max(data_range / 2000.0, 1e-9)
    slider_max_width = data_range

    slider_step = data_range / 2000.0

    default_level = (data_max + data_min) / 2.0
    default_width = data_range

    json_data = json.dumps(data.tolist())

    html_content = HTML_TEMPLATE.format(
        file_path=os.path.basename(npy_path),
        data_shape=str(data.shape),
        data_min=data_min,
        data_max=data_max,
        data_mean=data_mean,
        data_std=data_std,
        img_width=img_width,
        img_height=img_height,
        json_data=json_data,
        slider_min_level=slider_min_level,
        slider_max_level=slider_max_level,
        slider_min_width=slider_min_width,
        slider_max_width=slider_max_width,
        slider_step=f"{slider_step:.8f}",
        default_level=f"{default_level:.6f}",
        default_width=f"{default_width:.6f}",
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n--- 操作成功 ---")
    print(f"交互式查看器已保存到: {output_path}")
    print("请在您的网页浏览器中打开此文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从一个2D .npy 文件创建一个具备专业窗宽窗位调节功能的HTML查看器。")
    parser.add_argument("npy_file", help="输入的 .npy 文件路径。")
    parser.add_argument("-o", "--output", help="输出的 HTML 文件路径。默认为 'dicom_style_viewer.html'。")

    args = parser.parse_args()

    output_filename = args.output if args.output else "dicom_style_viewer.html"

    create_viewer(args.npy_file, output_filename)