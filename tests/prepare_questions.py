import os
import json
import glob

def prepare_questions():
    questions_dir = os.path.join("rag", "knowledge", "Questions")
    output_file = os.path.join("tests", "questions.json")
    
    all_questions = []
    
    # Pattern to match all txt files in the questions directory
    file_pattern = os.path.join(questions_dir, "*.txt")
    files = glob.glob(file_pattern)
    
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                # Skip headers and empty lines
                if not line:
                    continue
                if line.endswith("Questions"):
                    continue
                # Add question if it's not empty and doesn't look like a header
                if line:
                    all_questions.append(line)
    
    # Remove duplicates if any
    all_questions = list(set(all_questions))
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=4)
    
    print(f"Successfully extracted {len(all_questions)} questions to {output_file}")

if __name__ == "__main__":
    prepare_questions()
