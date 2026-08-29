"""Minimal smoke test: all .py compile, spec parses, MJCF compiles, HDF5 readable."""
import ast, json, glob, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
fails = []
for f in glob.glob("**/*.py", recursive=True):
    if "__pycache__" in f: continue
    try: compile(open(f, encoding="utf-8").read(), f, "exec")
    except SyntaxError as e: fails.append(f"{f}: {e}")
for f in glob.glob("**/scene_spec.json", recursive=True):
    try: json.load(open(f, encoding="utf-8"))
    except Exception as e: fails.append(f"{f}: {e}")
import mujoco
for f in glob.glob("**/generated/*.xml", recursive=True):
    try: mujoco.MjModel.from_xml_path(f)
    except Exception as e: fails.append(f"{f}: {e}")
for f in glob.glob("data/**/*.hdf5", recursive=True):
    try:
        import h5py
        h5py.File(f, "r").close()
    except Exception as e: fails.append(f"{f}: {e}")
if fails:
    print("SMOKE_FAIL"); [print(" ", x) for x in fails]; sys.exit(1)
print(f"SMOKE_OK ({len(glob.glob('**/*.py', recursive=True))} py, "
      f"{len(glob.glob('**/scene_spec.json', recursive=True))} spec, "
      f"{len(glob.glob('**/generated/*.xml', recursive=True))} xml, "
      f"{len(glob.glob('data/**/*.hdf5', recursive=True))} hdf5)")
