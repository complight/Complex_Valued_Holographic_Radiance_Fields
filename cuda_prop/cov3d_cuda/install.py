import os
import subprocess
import sys
'''
Main setup file for the CUDA extension.
'''
def main():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # First, try to setup GLM if needed
    try:
        from setup_glm import setup_glm
        glm_dir = setup_glm()
        print(f"GLM setup complete at: {glm_dir}")
    except Exception as e:
        print(f"Warning: Error setting up GLM - {e}")
        print("You may need to manually install GLM or adjust paths")
    
    # Build the extension in place
    build_cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    print(f"Running: {' '.join(build_cmd)}")
    
    try:
        subprocess.check_call(build_cmd, cwd=current_dir)
        print("\nBuild successful! The CUDA extension is now available for import.")
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
