from run import TextDetection
def main():
    text_detection = TextDetection(
        det_algorithm="DB",
        det_model_dir="/home/ayush/Desktop/English_handwritten/services/global",
        image_dir="/home/ayush/Desktop/English_handwritten/images/test_data",
        use_gpu=False
    )

    try:
        detection_results = text_detection.run_text_detection()
        for result in detection_results:
            print(result)
    except RuntimeError as e:
        print(str(e))

if __name__ == "__main__":
    main()