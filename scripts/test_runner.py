import os
import subprocess
import json

def run_test(build_path):
    print(f"🎬 Starting testing for: {build_path}")
    test_dir = os.path.join(build_path, "tests")
    if not os.path.exists(test_dir):
        print(f"❌ No tests found in {build_path}")
        return
    
    results = {}
    for test_file in os.listdir(test_dir):
        if test_file.endswith(".py"):
            print(f"🧪 Running {test_file}...")
            # Use the virtualenv python
            cmd = f"source .venv/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)/mnpp/core/fused-ops-mlx/src && python3 {os.path.join(test_dir, test_file)}"
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable="/bin/zsh")
            stdout, stderr = process.communicate()
            
            output = stdout.decode()
            print(output)
            results[test_file] = output

    return results

if __name__ == "__main__":
    # Test all 11 builds in order
    builds = [
        "builds/speed-demon-v1",
        "builds/memory-whisperer-v1",
        "builds/multimodal-sonic-v1",
        "builds/image-artisan-v1",
        "builds/training-titan-v1",
        "builds/agent-architect-v1",
        "builds/video-alchemist-v1",
        "builds/rag-radar-v1",
        "builds/code-commander-v1",
        "builds/bio-ml-architect-v1",
        "builds/tiny-titan-v1"
    ]
    
    global_report = {}
    for b in builds:
        res = run_test(b)
        global_report[b] = res
        
    # Save global report
    with open("experiments/GLOBAL_PASS_REPORT.json", "w") as f:
        json.dump(global_report, f, indent=4)
    print("🏁 Sequential testing complete. Report saved to experiments/GLOBAL_PASS_REPORT.json")
