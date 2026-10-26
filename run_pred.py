import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"🚀 开始执行: {script_name}")
    print(f"{'='*50}")
    
    # 获取当前主目录路径 (E:\bye\code\chazhi_7303)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 复制当前的环境变量，并将主目录加入 PYTHONPATH
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{current_dir};{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = current_dir

    # 运行脚本，传入修改后的环境变量
    result = subprocess.run([sys.executable, script_name], env=env)
    
    # 检查是否报错挂掉
    if result.returncode != 0:
        print(f"❌ 运行 {script_name} 时发生错误，停止后续执行。")
        sys.exit(result.returncode)
    else:
        print(f"✅ {script_name} 执行完毕！\n")

if __name__ == "__main__":
    # 把你想按顺序执行的脚本名放到这里
    scripts_to_run = [
        # "run_predrnn\pred_tx_cnn-lstm.py", 
        # "run_predrnn\pred_tx_cnn.py",
        # "run_predrnn\pred_tx_convlstm.py", 
        # "run_predrnn\pred_tx_lstm.py",
        # "run_predrnn\pred_tx_predrnn.py", 
        # "run_predrnn\pred_tx_simvp.py",       
        "run_convlstm3/pred_tx.py",
        "run_convlstm3/pred_tx_cov1.py",
        "run_convlstm3/pred_tx_cov2.py",
        "run_convlstm3/pred_tx_cov3.py",
        "run_convlstm3/pred_ty.py",
        "run_convlstm3/pred_ty_cov1.py",
        "run_convlstm3/pred_ty_cov2.py",
        "run_convlstm3/pred_ty_cov3.py",
        "run_convlstm3/pred_tz.py",
        "run_convlstm3/pred_tz_cov1.py",
        "run_convlstm3/pred_tz_cov2.py",
        "run_convlstm3/pred_tz_cov3.py"
    ]
    
    for script in scripts_to_run:
        run_script(script)
        
    print("🎉 所有任务均已执行完毕！")