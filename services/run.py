import subprocess
import re

class TextDetection:
    def __init__(self, det_algorithm="DB", det_model_dir=" ",
                 image_dir=" ", use_gpu=False):
        self.det_algorithm = det_algorithm
        self.det_model_dir = det_model_dir
        self.image_dir = image_dir
        self.use_gpu = use_gpu

    def run_text_detection(self):
        command = [
            "python3",
            r"/home/ayush/Desktop/English_handwritten/services/tools/infer/predict_det.py",
            f"--det_algorithm={self.det_algorithm}",
            f"--det_model_dir={self.det_model_dir}",
            f"--image_dir={self.image_dir}",
            f"--use_gpu={self.use_gpu}"
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return self.parse_output(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"An error occurred while running the command: {e.stderr}") from e

    def parse_output(self, output):
        results = []
        bbox_pattern = re.compile(r"(\w+\.\w+)\s+\[\[")

        current_filename = None
        current_bboxes = ""

        for line in output.splitlines():
            print(f"Processing line: {line}")
            match = bbox_pattern.search(line)
            if match:
                print(f"Match found: {match.groups()}")
                if current_filename:
                    try:
                        bboxes = eval(current_bboxes)
                        results.append({"filename": current_filename, "bounding_boxes": bboxes})
                    except SyntaxError:
                        print("SyntaxError during final eval")
                current_filename = match.group(1)
                current_bboxes = line[line.index("[["):]
            elif current_filename:
                current_bboxes += line
                try:
                    bboxes = eval(current_bboxes)
                    results.append({"filename": current_filename, "bounding_boxes": bboxes})
                    current_filename = None
                    current_bboxes = ""
                except SyntaxError:
                    continue

        if current_filename:
            try:
                bboxes = eval(current_bboxes)
                results.append({"filename": current_filename, "bounding_boxes": bboxes})
            except SyntaxError:
                print("SyntaxError during final eval of last file")

        print(f"Final results :\n {results}")
        return results


