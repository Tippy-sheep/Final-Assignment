#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
北京PM2.5智能监测系统 - 一键运行脚本
功能：数据预处理、模型训练、边缘服务器启动
"""

import subprocess
import sys
import os
import time
import webbrowser
import argparse
import platform
from datetime import datetime

# ============================================================
# 配置
# ============================================================
MODEL_TYPE = 'GRU'  # 可选: 'LSTM', 'GRU', 'RNN'
AUTO_OPEN_BROWSER = True  # 是否自动打开浏览器
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765

# 颜色输出
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Windows 下禁用颜色
if platform.system() == 'Windows':
    Colors.HEADER = ''
    Colors.BLUE = ''
    Colors.CYAN = ''
    Colors.GREEN = ''
    Colors.YELLOW = ''
    Colors.RED = ''
    Colors.ENDC = ''
    Colors.BOLD = ''
    Colors.UNDERLINE = ''

def print_header(text):
    """打印标题"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

def print_step(step, text):
    """打印步骤"""
    print(f"{Colors.CYAN}[步骤 {step}]{Colors.ENDC} {text}")

def print_success(text):
    """打印成功信息"""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")

def print_error(text):
    """打印错误信息"""
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")

def print_warning(text):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {text}")

def print_info(text):
    """打印信息"""
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {text}")

def run_command(command, description):
    """运行命令并检查结果"""
    print(f"\n{Colors.CYAN}>> 执行: {command}{Colors.ENDC}")
    print(f"   {description}...")

    try:
        # 在Windows上使用sys.executable确保使用正确的Python解释器
        if platform.system() == 'Windows':
            result = subprocess.run(
                [sys.executable] + command.split()[1:],
                capture_output=False,
                text=True,
                check=False
            )
        else:
            result = subprocess.run(
                command.split(),
                capture_output=False,
                text=True,
                check=False
            )

        if result.returncode == 0:
            print_success(f"{description}完成")
            return True
        else:
            print_error(f"{description}失败 (退出码: {result.returncode})")
            return False
    except Exception as e:
        print_error(f"{description}异常: {e}")
        return False

def check_dependencies():
    """检查依赖包"""
    print_step("0", "检查依赖包")

    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'websockets',
        'asyncio'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} 已安装")
        except ImportError:
            print_warning(f"{package} 未安装")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n{Colors.YELLOW}缺少以下依赖包: {', '.join(missing_packages)}{Colors.ENDC}")
        install = input("\n是否自动安装？(y/n): ").lower()
        if install == 'y':
            for package in missing_packages:
                print_info(f"正在安装 {package}...")
                if package == 'asyncio':
                    package = 'asyncio'  # asyncio 是内置模块
                elif package == 'websockets':
                    subprocess.run([sys.executable, '-m', 'pip', 'install', 'websockets'])
                else:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package])
            print_success("依赖包安装完成")
            return True
        else:
            print_warning("请手动安装缺失的依赖包")
            return False

    return True

def check_data_files():
    """检查数据文件"""
    print_step("1", "检查数据文件")

    required_files = [
        'data/raw_data/BeijingPM20100101_20151231.csv',
        'data_preprocess.py'
    ]

    missing_files = []

    for file in required_files:
        if os.path.exists(file):
            print_success(f"{file} 存在")
        else:
            print_warning(f"{file} 不存在")
            missing_files.append(file)

    if missing_files:
        print_error("缺少必要的数据文件")
        print_info("请确保:")
        print("  1. 原始数据文件在 data/raw_data/ 目录下")
        print("  2. 文件名为 BeijingPM20100101_20151231.csv")
        return False

    return True

def run_data_preprocessing():
    """运行数据预处理"""
    print_step("2", "数据预处理")

    if not os.path.exists('data_preprocess.py'):
        print_error("找不到 data_preprocess.py 文件")
        return False

    return run_command('python data_preprocess.py', "数据预处理")

def run_model_training():
    """运行模型训练"""
    print_step("3", f"训练 {MODEL_TYPE} 模型")

    # 检查 train_model.py 是否存在
    if not os.path.exists('train_model.py'):
        print_error("找不到 train_model.py 文件")
        return False

    # 修改 train_model.py 中的 MODEL_TYPE
    print_info(f"设置模型类型为: {MODEL_TYPE}")

    try:
        with open('train_model.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换 MODEL_TYPE
        import re
        content = re.sub(r"MODEL_TYPE\s*=\s*'[^']*'", f"MODEL_TYPE = '{MODEL_TYPE}'", content)

        with open('train_model.py', 'w', encoding='utf-8') as f:
            f.write(content)

        print_success(f"已设置 train_model.py 中的 MODEL_TYPE = '{MODEL_TYPE}'")
    except Exception as e:
        print_warning(f"无法自动修改训练脚本: {e}")

    # 运行训练
    return run_command('python train_model.py', f"{MODEL_TYPE} 模型训练")

def check_model_weights():
    """检查模型权重文件"""
    print_step("4", "检查模型权重")

    weights_file = f'models/deploy_{MODEL_TYPE.lower()}_weights.pth'

    if os.path.exists(weights_file):
        file_size = os.path.getsize(weights_file) / 1024  # KB
        print_success(f"{weights_file} 存在 ({file_size:.1f} KB)")
        return True
    else:
        print_warning(f"{weights_file} 不存在")
        print_info("请先训练模型")
        return False

def start_edge_server():
    """启动边缘服务器"""
    print_step("5", "启动边缘服务器")

    # 检查 edge_server.py 是否存在
    if not os.path.exists('edge_server.py'):
        print_error("找不到 edge_server.py 文件")
        return None

    # 修改 edge_server.py 中的 MODEL_TYPE
    print_info(f"设置服务器模型类型为: {MODEL_TYPE}")

    try:
        with open('edge_server.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换 MODEL_TYPE
        import re
        content = re.sub(r"MODEL_TYPE\s*=\s*'[^']*'", f"MODEL_TYPE = '{MODEL_TYPE}'", content)

        with open('edge_server.py', 'w', encoding='utf-8') as f:
            f.write(content)

        print_success(f"已设置 edge_server.py 中的 MODEL_TYPE = '{MODEL_TYPE}'")
    except Exception as e:
        print_warning(f"无法自动修改服务器脚本: {e}")

    print_info(f"启动 WebSocket 服务器在 ws://{SERVER_HOST}:{SERVER_PORT}")
    print_info("按 Ctrl+C 停止服务器")

    # 启动服务器进程
    try:
        # 在后台启动服务器
        if platform.system() == 'Windows':
            server_process = subprocess.Popen(
                [sys.executable, 'edge_server.py'],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            server_process = subprocess.Popen(
                [sys.executable, 'edge_server.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

        # 等待服务器启动
        time.sleep(3)

        # 检查进程是否还在运行
        if server_process.poll() is None:
            print_success("服务器启动成功")
            return server_process
        else:
            print_error("服务器启动失败")
            return None

    except Exception as e:
        print_error(f"启动服务器异常: {e}")
        return None

def open_web_interface():
    """打开网页界面"""
    print_step("6", "打开网页界面")

    index_file = os.path.abspath('index.html')

    if not os.path.exists(index_file):
        print_error(f"找不到 {index_file}")
        return False

    file_url = f'file:///{index_file}'

    print_info(f"打开网页: {file_url}")

    try:
        webbrowser.open(file_url)
        print_success("浏览器已打开")
        return True
    except Exception as e:
        print_error(f"无法打开浏览器: {e}")
        print_info(f"请手动打开: {index_file}")
        return False

def show_menu():
    """显示菜单"""
    print_header("北京PM2.5智能监测系统")
    print(f"{Colors.BOLD}请选择操作:{Colors.ENDC}")
    print("  1. 完整运行 (预处理 → 训练 → 启动服务器)")
    print("  2. 仅训练模型")
    print("  3. 仅启动服务器")
    print("  4. 切换模型类型")
    print("  5. 检查环境")
    print("  6. 退出")
    print()

    choice = input(f"{Colors.CYAN}请输入选项 (1-6): {Colors.ENDC}").strip()
    return choice

def switch_model():
    """切换模型类型"""
    global MODEL_TYPE

    print_header("切换模型类型")
    print(f"当前模型: {Colors.GREEN}{MODEL_TYPE}{Colors.ENDC}")
    print("\n可选模型:")
    print("  1. LSTM (长短期记忆网络)")
    print("  2. GRU (门控循环单元)")
    print("  3. RNN (标准循环神经网络)")
    print()

    choice = input(f"{Colors.CYAN}请选择模型 (1-3): {Colors.ENDC}").strip()

    if choice == '1':
        MODEL_TYPE = 'LSTM'
    elif choice == '2':
        MODEL_TYPE = 'GRU'
    elif choice == '3':
        MODEL_TYPE = 'RNN'
    else:
        print_warning("无效选择，保持当前模型")
        return

    print_success(f"已切换到模型: {MODEL_TYPE}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='北京PM2.5智能监测系统 - 一键运行')
    parser.add_argument('--model', type=str, default='GRU',
                       choices=['LSTM', 'GRU', 'RNN'],
                       help='选择模型类型 (默认: GRU)')
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='跳过数据预处理')
    parser.add_argument('--skip-train', action='store_true',
                       help='跳过模型训练')
    parser.add_argument('--no-browser', action='store_true',
                       help='不自动打开浏览器')

    args = parser.parse_args()

    # 更新全局变量
    global MODEL_TYPE, AUTO_OPEN_BROWSER
    MODEL_TYPE = args.model
    if args.no_browser:
        AUTO_OPEN_BROWSER = False

    # 如果没有命令行参数，显示菜单
    if len(sys.argv) == 1:
        while True:
            choice = show_menu()

            if choice == '1':
                # 完整运行
                if not check_dependencies():
                    continue
                if not check_data_files():
                    continue
                if not run_data_preprocessing():
                    continue
                if not run_model_training():
                    continue
                if not check_model_weights():
                    continue

                server_process = start_edge_server()
                if server_process and AUTO_OPEN_BROWSER:
                    open_web_interface()

                if server_process:
                    print_info("服务器正在运行，按 Enter 键停止...")
                    input()
                    server_process.terminate()
                    print_success("服务器已停止")
                break

            elif choice == '2':
                # 仅训练
                if not check_dependencies():
                    continue
                run_model_training()
                input("\n按 Enter 键继续...")

            elif choice == '3':
                # 仅启动服务器
                if not check_model_weights():
                    continue
                server_process = start_edge_server()
                if server_process and AUTO_OPEN_BROWSER:
                    open_web_interface()
                if server_process:
                    input("\n按 Enter 键停止服务器...")
                    server_process.terminate()
                    print_success("服务器已停止")
                break

            elif choice == '4':
                # 切换模型
                switch_model()

            elif choice == '5':
                # 检查环境
                check_dependencies()
                check_data_files()
                check_model_weights()
                input("\n按 Enter 键继续...")

            elif choice == '6':
                print_success("再见！")
                break
            else:
                print_error("无效选项，请重新选择")

    else:
        # 命令行模式
        print_header(f"北京PM2.5智能监测系统 - {MODEL_TYPE}模型")

        start_time = datetime.now()

        # 检查依赖
        if not check_dependencies():
            sys.exit(1)

        # 数据预处理
        if not args.skip_preprocess:
            if not run_data_preprocessing():
                sys.exit(1)
        else:
            print_info("跳过数据预处理")

        # 模型训练
        if not args.skip_train:
            if not run_model_training():
                sys.exit(1)
        else:
            print_info("跳过模型训练")

        # 检查权重
        if not check_model_weights():
            sys.exit(1)

        # 启动服务器
        server_process = start_edge_server()
        if not server_process:
            sys.exit(1)

        # 打开浏览器
        if AUTO_OPEN_BROWSER:
            open_web_interface()

        # 显示运行信息
        elapsed = (datetime.now() - start_time).total_seconds()
        print_success(f"系统启动完成！耗时 {elapsed:.1f} 秒")
        print_info(f"服务器地址: ws://{SERVER_HOST}:{SERVER_PORT}")
        print_info(f"模型类型: {MODEL_TYPE}")
        print_info("按 Ctrl+C 停止服务器")

        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n")
            print_warning("正在停止服务器...")
            server_process.terminate()
            print_success("服务器已停止")

if __name__ == "__main__":
    main()