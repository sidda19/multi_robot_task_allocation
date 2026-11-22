import os
import shutil
import zipfile
import subprocess
import sys

def package_submission():
    base_dir = os.path.abspath(".")
    sub_dir = os.path.join(base_dir, "submission_corrected")
    zip_path = os.path.join(base_dir, "MRTA_LC_Group45_Corrected_Submission.zip")

    print(f"Cleaning previous submission directory if exists: {sub_dir}")
    if os.path.exists(sub_dir):
        shutil.rmtree(sub_dir)
    os.makedirs(sub_dir, exist_ok=True)

    # Directories to copy
    dirs_to_copy = [
        "src",
        "tests",
        "models",
        "results",
        "report",
        "scripts",
        "screenshots"
    ]

    # Files to copy
    files_to_copy = [
        "config.py",
        "run_experiments.py",
        "run_member1_member2_demo.py",
        "README.md",
        "requirements.txt",
        ".gitignore",
        "SUBMISSION_CHECKLIST.md",
        "REPRODUCIBILITY.md",
        "LIMITATIONS.md",
        
        "
        "FAI_project_idea_report.pdf",
        "feedback.txt"
    ]

    print("Copying root files...")
    for f in files_to_copy:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            dst = os.path.join(sub_dir, f)
            shutil.copy2(src, dst)
            print(f"  Copied: {f}")
        else:
            print(f"  Notice: {f} not found or omitted.")

    print("Copying directories (excluding __pycache__ and temp files)...")
    def ignore_patterns(path, names):
        return [n for n in names if n in ('__pycache__', '.git', '.pytest_cache') or n.endswith(('.pyc', '.pyo', '.tmp'))]

    for d in dirs_to_copy:
        src = os.path.join(base_dir, d)
        if os.path.exists(src):
            dst = os.path.join(sub_dir, d)
            shutil.copytree(src, dst, ignore=ignore_patterns)
            print(f"  Copied dir: {d}")
        else:
            print(f"  Warning: directory {d} not found!")

    # Create Zip Archive
    print(f"\nCreating ZIP archive: {zip_path}")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(sub_dir):
            for file in files:
                abs_file = os.path.join(root, file)
                rel_path = os.path.relpath(abs_file, sub_dir)
                # Store with top-level folder name inside zip
                arcname = os.path.join("MRTA_LC_Group45_Corrected_Submission", rel_path)
                zipf.write(abs_file, arcname)

    zip_size = os.path.getsize(zip_path)
    zip_size_mb = zip_size / (1024 * 1024)
    print(f"ZIP Archive successfully created!")
    print(f"Path: {zip_path}")
    print(f"Size: {zip_size:,} bytes ({zip_size_mb:.2f} MB)")
    assert zip_size_mb < 20.0, f"Error: ZIP file size ({zip_size_mb:.2f} MB) exceeds 20 MB limit!"
    print("Archive size is strictly within the 20 MB limit.")

    # Verify extraction in temporary directory
    temp_dir = os.path.join(base_dir, "temp_verify_submission")
    print(f"\nVerifying extraction into fresh temporary directory: {temp_dir}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(temp_dir)
    print("Extracted archive successfully.")

    extracted_root = os.path.join(temp_dir, "MRTA_LC_Group45_Corrected_Submission")
    print(f"Running unit test suite inside extracted package: {extracted_root}")

    # Run unittest inside extracted directory with current environment
    test_env = os.environ.copy()
    libs_dir = os.path.join(base_dir, "libs")
    if os.path.exists(libs_dir):
        test_env["PYTHONPATH"] = libs_dir + os.pathsep + test_env.get("PYTHONPATH", "")

    res = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=extracted_root,
        env=test_env,
        capture_output=True,
        text=True
    )
    print(res.stdout)
    if res.returncode != 0:
        print("STDERR:", res.stderr)
        raise RuntimeError("Verification tests FAILED in extracted submission!")
    print("All tests PASSED in extracted submission directory!")

    # Run mathematical/result consistency audit inside extracted package.
    print("Running mathematical and result consistency audit...")
    res_audit = subprocess.run(
        [sys.executable, "scripts/audit_math.py"],
        cwd=extracted_root,
        env=test_env,
        capture_output=True,
        text=True
    )
    print(res_audit.stdout)
    if res_audit.returncode != 0:
        print("Audit STDERR:", res_audit.stderr)
        raise RuntimeError("Mathematical/result consistency audit FAILED in extracted submission!")

    # Run demo inside extracted directory
    print("Running smoke-test demo inside extracted package...")
    res_demo = subprocess.run(
        [sys.executable, "run_member1_member2_demo.py"],
        cwd=extracted_root,
        env=test_env,
        capture_output=True,
        text=True
    )
    if res_demo.returncode != 0:
        print("Demo STDERR:", res_demo.stderr)
        raise RuntimeError("Demo execution FAILED in extracted submission!")
    print("Demo smoke-test executed cleanly!")

    # Clean up temp_verify_submission
    print("Cleaning up temporary verification directory...")
    shutil.rmtree(temp_dir)
    print("Verification complete and cleanup done!")

if __name__ == "__main__":
    package_submission()
