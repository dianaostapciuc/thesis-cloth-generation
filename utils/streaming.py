def iterfile(path: str, chunk_size: int = 1024 * 1024 * 4):
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk
