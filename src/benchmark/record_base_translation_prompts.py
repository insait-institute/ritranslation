import os
import pyinputplus as pyip

def get_multiline_input():
    print("Enter your prompt text (multiline). You can enter multiple lines, press Enter after each line.")
    print("Type 'END NOW' (in uppercase, without quotes) on a new line when you are finished:")
    lines = []
    while True:
        line = input()
        if line.strip() == "END NOW":
            break
        lines.append(line)
    return "\n".join(lines)

def main():
    
    files = []
    while True:
        
        record_choice = pyip.inputYesNo("Do you want to record a prompt and save it? (yes/no): ")
        if record_choice == "no":
            print(f"Exiting the prompt recorder. Following files were saved: {files}")
            break

        prompt_text = get_multiline_input()

        file_name = pyip.inputStr("Enter the filename for this prompt (without the extension): ").strip().replace(" ", "_")
        if not file_name:
            print("No file name provided. Skipping saving this prompt.")
            continue

        data_folder = "src/benchmark/prompts/mq_base_translation_prompts"
        os.makedirs(data_folder, exist_ok=True)

        file_path = os.path.join(data_folder, file_name + ".txt")
        files.append(file_path)
        
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(prompt_text)
            print(f"Prompt saved in '{file_path}'!")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")

if __name__ == "__main__":
    main()