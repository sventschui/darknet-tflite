def parse_darknet_cfg(path):
    net = None  # the first [net] block
    layers = []  # all other layer blocks
    current = None
    seen_net = False

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()

            # ignore blanks and comments
            if not line or line.startswith("#"):
                continue

            # start of a new block
            if line.startswith("[") and line.endswith("]"):
                # finalize previous block
                if current is not None:
                    if current["type"] == "net" and not seen_net:
                        net = current["params"]
                        seen_net = True
                    else:
                        layers.append(current)

                # create new block
                block_type = line[1:-1].strip()
                current = {"type": block_type, "params": {}}
                continue

            # parameter line
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key in ["anchors", "mask"]:
                    value = [int(anchor.strip()) for anchor in value.split(",")]
                elif key in ["width", "height", "classes", "num"]:
                    value = int(value)
                elif key in [
                    "jitter",
                    "scale_x_y",
                    "cls_normalizer",
                    "iou_normalizer",
                    "ignore_thresh",
                    "truth_thresh",
                    "random",
                    "resize",
                    "beta_nms",
                ]:
                    value = float(value)

                current["params"][key] = value
            else:
                # flag-style keys with no assigned value
                current["params"][line] = None

    # finalize last block
    if current is not None:
        if current["type"] == "net" and not seen_net:
            net = current["params"]
        else:
            layers.append(current)

    return net, layers
