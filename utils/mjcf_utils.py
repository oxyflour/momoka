import os
import shutil
import tempfile
from typing import Optional


def mjcf_to_urdf(mjcf_path: str, urdf_path: Optional[str] = None) -> str:
    """Convert MuJoCo MJCF file to URDF format.

    Args:
        mjcf_path: Path to the MJCF XML file.
        urdf_path: Optional path to save the URDF file. If None, creates a temp file.

    Returns:
        Path to the generated URDF file.
    """
    try:
        import mjcf2urdf
    except ImportError:
        raise ImportError(
            "mjcf2urdf is required for MJCF to URDF conversion. "
            "Install it with: pip install mjcf2urdf"
        )

    if urdf_path is None:
        temp_dir = tempfile.mkdtemp()
        base_name = os.path.splitext(os.path.basename(mjcf_path))[0]
        urdf_path = os.path.join(temp_dir, f"{base_name}.urdf")

    if hasattr(mjcf2urdf, "mjcf2urdf"):
        urdf = mjcf2urdf.mjcf2urdf(mjcf_path)
        with open(urdf_path, "w") as f:
            f.write(urdf)
    elif hasattr(mjcf2urdf, "convert_mjcf_to_urdf"):
        import inspect
        from glob import glob

        fn = mjcf2urdf.convert_mjcf_to_urdf
        params = inspect.signature(fn).parameters
        if "output_path" in params:
            target_path = urdf_path if urdf_path.lower().endswith(".urdf") else None
            output_dir = (
                os.path.dirname(urdf_path)
                if target_path
                else os.path.abspath(urdf_path)
            )
            existing = set(glob(os.path.join(output_dir, "*.urdf")))
            fn(mjcf_path, output_dir)
            candidates = [
                path
                for path in glob(os.path.join(output_dir, "*.urdf"))
                if path not in existing
            ]
            if not candidates:
                raise RuntimeError("mjcf2urdf did not produce a URDF file.")
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            generated = candidates[0]
            if target_path:
                if os.path.abspath(generated) != os.path.abspath(target_path):
                    shutil.copyfile(generated, target_path)
                urdf_path = target_path
            else:
                urdf_path = generated
        else:
            urdf = fn(mjcf_path)
            with open(urdf_path, "w") as f:
                f.write(urdf)
    else:
        raise AttributeError(
            "mjcf2urdf does not expose a known conversion function."
        )

    return urdf_path


def mjcf_ctrl_range(mjcf_path: str):
    """Load actuator ctrlrange from an MJCF file."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    return model.actuator_ctrlrange.copy()
