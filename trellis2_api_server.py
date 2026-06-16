#!/usr/bin/env python3
"""Small Meshy-like local HTTP API for TRELLIS.2 image-to-3D generation."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shutil
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse
from urllib.request import urlopen


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7862
DEFAULT_MODEL = "microsoft/TRELLIS.2-4B"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "tmp" / "api"

PIPELINE = None
TASKS: dict[str, dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()
GENERATION_SEMAPHORE: threading.Semaphore
SERVER_CONFIG: argparse.Namespace


def _now() -> float:
    return time.time()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    body = handler.rfile.read(content_length)
    return json.loads(body.decode("utf-8"))


def _task_payload(task: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": task["id"],
        "asset_id": task["id"],
        "status": task["status"],
        "created_at": task["created_at"],
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "model_urls": task.get("model_urls", {}),
        "thumbnail_url": task.get("thumbnail_url"),
        "progress": task.get("progress", 0),
    }
    if task.get("error"):
        payload["task_error"] = {"message": task["error"]}
    return payload


def _update_task(task_id: str, **updates: Any) -> None:
    with TASKS_LOCK:
        if task_id in TASKS:
            TASKS[task_id].update(updates)


def _get_task(task_id: str) -> dict[str, Any] | None:
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        return dict(task) if task else None


def _task_dir(task_id: str) -> Path:
    return Path(SERVER_CONFIG.output_dir).resolve() / task_id


def _public_base_url(handler_or_base: BaseHTTPRequestHandler | str | None = None) -> str:
    if isinstance(handler_or_base, str) and handler_or_base:
        return handler_or_base.rstrip("/")
    if handler_or_base is not None:
        host = handler_or_base.headers.get("Host") or f"{SERVER_CONFIG.host}:{SERVER_CONFIG.port}"
        return f"http://{host}"
    return f"http://{SERVER_CONFIG.host}:{SERVER_CONFIG.port}"


def _guess_extension(content_type: str | None, fallback: str = ".png") -> str:
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ".jpg" if ext == ".jpe" else ext
    return fallback


def _save_image_input(image_url: str, task_id: str) -> Path:
    output_dir = _task_dir(task_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image_url.startswith("data:"):
        header, encoded = image_url.split(",", 1)
        content_type = header[5:].split(";")[0] if ";" in header else "image/png"
        image_path = output_dir / f"input{_guess_extension(content_type)}"
        image_path.write_bytes(base64.b64decode(encoded))
        return image_path

    parsed = urlparse(image_url)
    if parsed.scheme in {"http", "https"}:
        with urlopen(image_url, timeout=60) as response:
            content_type = response.headers.get("Content-Type")
            image_path = output_dir / f"input{_guess_extension(content_type)}"
            image_path.write_bytes(response.read())
            return image_path

    local_path = Path(os.path.expanduser(image_url)).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_url}")

    suffix = local_path.suffix or ".png"
    image_path = output_dir / f"input{suffix}"
    shutil.copy2(local_path, image_path)
    return image_path


def _get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        PIPELINE = Trellis2ImageTo3DPipeline.from_pretrained(SERVER_CONFIG.model)
        PIPELINE.cuda()
    return PIPELINE


def _run_trellis_generation(task_id: str, request_payload: dict[str, Any], public_base_url: str) -> None:
    _update_task(task_id, status="PENDING", progress=0)
    with GENERATION_SEMAPHORE:
        try:
            _update_task(task_id, status="RUNNING", started_at=_now(), progress=5)
            image_path = _save_image_input(request_payload["image_url"], task_id)
            _update_task(task_id, thumbnail_url=f"{public_base_url}/files/{task_id}/{quote(image_path.name)}")

            from PIL import Image
            import torch
            import o_voxel

            pipeline = _get_pipeline()
            image = Image.open(image_path).convert("RGBA")
            image = pipeline.preprocess_image(image)

            resolution = str(request_payload.get("resolution", SERVER_CONFIG.resolution))
            pipeline_type = {
                "512": "512",
                "1024": "1024_cascade",
                "1536": "1536_cascade",
            }.get(resolution)
            if pipeline_type is None:
                raise ValueError("resolution must be one of 512, 1024, or 1536")

            seed = request_payload.get("seed")
            run_kwargs = {
                "preprocess_image": False,
                "pipeline_type": pipeline_type,
                "return_latent": True,
            }
            if seed is not None:
                run_kwargs["seed"] = int(seed)

            for request_key, sampler_key in [
                ("sparse_structure_sampler_params", "sparse_structure_sampler_params"),
                ("shape_slat_sampler_params", "shape_slat_sampler_params"),
                ("tex_slat_sampler_params", "tex_slat_sampler_params"),
            ]:
                params = request_payload.get(request_key)
                if isinstance(params, dict):
                    run_kwargs[sampler_key] = params

            _update_task(task_id, progress=20)
            outputs, latents = pipeline.run(image, **run_kwargs)
            mesh = outputs[0]
            mesh.simplify(16777216)

            _, _, grid_size = latents
            _update_task(task_id, progress=75)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=pipeline.pbr_attr_layout,
                grid_size=grid_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(request_payload.get("decimation_target", SERVER_CONFIG.decimation_target)),
                texture_size=int(request_payload.get("texture_size", SERVER_CONFIG.texture_size)),
                remesh=bool(request_payload.get("remesh", True)),
                remesh_band=int(request_payload.get("remesh_band", 1)),
                remesh_project=int(request_payload.get("remesh_project", 0)),
                use_tqdm=bool(request_payload.get("use_tqdm", True)),
            )

            glb_path = _task_dir(task_id) / "model.glb"
            glb.export(glb_path, extension_webp=True)
            torch.cuda.empty_cache()

            model_url = f"{public_base_url}/files/{task_id}/model.glb"
            _update_task(
                task_id,
                status="SUCCEEDED",
                finished_at=_now(),
                progress=100,
                model_urls={"glb": model_url},
                local_model_path=str(glb_path),
            )
        except Exception as exc:
            _update_task(
                task_id,
                status="FAILED",
                finished_at=_now(),
                progress=100,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )


class Trellis2APIHandler(BaseHTTPRequestHandler):
    server_version = "Trellis2MeshyLikeAPI/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        if not getattr(SERVER_CONFIG, "quiet", False):
            super().log_message(format, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/health":
            _json_response(
                self,
                200,
                {
                    "status": "ok",
                    "model_loaded": PIPELINE is not None,
                    "max_concurrent": SERVER_CONFIG.max_concurrent,
                },
            )
            return

        prefix = "/openapi/v1/image-to-3d/"
        if parsed.path.startswith(prefix):
            task_id = unquote(parsed.path[len(prefix):].strip("/"))
            task = _get_task(task_id)
            if not task:
                _json_response(self, 404, {"error": f"Unknown task: {task_id}"})
                return
            _json_response(self, 200, _task_payload(task))
            return

        if parsed.path.startswith("/files/"):
            parts = parsed.path.split("/", 3)
            if len(parts) != 4:
                self.send_error(404)
                return
            task_id = unquote(parts[2])
            filename = unquote(parts[3])
            file_path = (_task_dir(task_id) / filename).resolve()
            if _task_dir(task_id).resolve() not in file_path.parents or not file_path.exists():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") != "/openapi/v1/image-to-3d":
            self.send_error(404)
            return

        try:
            payload = _read_json(self)
            if not payload.get("image_url"):
                _json_response(self, 400, {"error": "Missing required field: image_url"})
                return

            task_id = str(uuid.uuid4())
            public_base_url = _public_base_url(self)
            task = {
                "id": task_id,
                "status": "PENDING",
                "created_at": _now(),
                "progress": 0,
                "model_urls": {},
                "thumbnail_url": None,
                "error": None,
            }
            with TASKS_LOCK:
                TASKS[task_id] = task

            worker = threading.Thread(
                target=_run_trellis_generation,
                args=(task_id, payload, public_base_url),
                daemon=True,
            )
            worker.start()
            _json_response(self, 200, {"result": task_id})
        except Exception as exc:
            _json_response(self, 500, {"error": f"{type(exc).__name__}: {exc}"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Meshy-like local TRELLIS.2 API server.")
    parser.add_argument("--host", default=os.getenv("TRELLIS2_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("TRELLIS2_PORT", DEFAULT_PORT)))
    parser.add_argument("--model", default=os.getenv("TRELLIS2_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=os.getenv("TRELLIS2_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--max-concurrent", type=int, default=int(os.getenv("TRELLIS2_MAX_CONCURRENT", "1")))
    parser.add_argument("--texture-size", type=int, default=int(os.getenv("TRELLIS2_TEXTURE_SIZE", "4096")))
    parser.add_argument("--resolution", default=os.getenv("TRELLIS2_RESOLUTION", "1024"), choices=["512", "1024", "1536"])
    parser.add_argument("--decimation-target", type=int, default=int(os.getenv("TRELLIS2_DECIMATION_TARGET", "1000000")))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    global SERVER_CONFIG, GENERATION_SEMAPHORE
    SERVER_CONFIG = parse_args()
    SERVER_CONFIG.output_dir = str(Path(SERVER_CONFIG.output_dir).expanduser().resolve())
    Path(SERVER_CONFIG.output_dir).mkdir(parents=True, exist_ok=True)
    GENERATION_SEMAPHORE = threading.Semaphore(max(1, SERVER_CONFIG.max_concurrent))

    server = ThreadingHTTPServer((SERVER_CONFIG.host, SERVER_CONFIG.port), Trellis2APIHandler)
    print(f"TRELLIS.2 API server listening at http://{SERVER_CONFIG.host}:{SERVER_CONFIG.port}")
    print(f"Meshy-like base URL: http://{SERVER_CONFIG.host}:{SERVER_CONFIG.port}/openapi/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
