import json
import re
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter
from pathlib import Path
import random
from tqdm import tqdm
import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_json_data(input_path: str) -> Dict[str, Any]:
    """Load the input JSON file with questions."""
    with open(input_path, 'r') as f:
        return json.load(f)

def load_existing_outputs(output_dir: Path, model_name: str, eval_type: str) -> Dict[str, Dict]:
    """Load existing outputs from the checkpoint file if it exists."""
    output_path = output_dir / f"{model_name.replace('/', '_')}_{eval_type}.json"
    
    if output_path.exists():
        with open(output_path, 'r') as f:
            data = json.load(f)
            # Create a mapping of question_id to results for easy lookup
            return {str(item["question_id"]): item for item in data}
    return {}

def save_checkpoint(data: List[Dict], output_dir: Path, model_name: str, eval_type: str):
    """Save current progress to JSON file."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = output_dir / f"{model_name.replace('/', '_')}_{eval_type}.json"
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_model_response(model, processor, formatted_query, max_retries: int = 3) -> str:
    """Get response from the Qwen Audio model with direct generation approach."""
    for attempt in range(max_retries):
        try:
            # Convert the formatted query to a conversation structure
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            ]
            
            # Add the audio and text to the conversation
            user_content = []
            if 'audio' in formatted_query[0]:
                audio_path = formatted_query[0]['audio']
                # Load audio using librosa
                import librosa
                audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
                user_content.append({"type": "audio", "audio": audio})
            
            # Add the text part
            if len(formatted_query) > 1 and 'text' in formatted_query[1]:
                user_content.append({"type": "text", "text": formatted_query[1]['text']})
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_content})
            
            # Process the conversation
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            
            # Get audios from the conversation
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(ele["audio"])
            
            # Prepare inputs
            inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate = processor.feature_extractor.sampling_rate)
            
            # Move inputs to the appropriate device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response
            generate_ids = model.generate(**inputs, max_length=1024)
            # generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            return response.strip().split("\nassistant\n")[-1].strip()
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}"
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            continue

def format_one_shot_prompt(question: str, question_id: str, audio_dir: str,
                           correct_answer: str, distractors: List[str]) -> Tuple[List, Dict[str, str]]:
    """Format the prompt for one-shot evaluation with Qwen Audio."""
    # Combine correct answer with distractors
    all_options = [correct_answer] + distractors
    random.shuffle(all_options)
    
    # Create mapping of letters to options for later evaluation
    letter_mapping = {}
    letters = ["A", "B", "C", "D", "E"][:len(all_options)]
    for i, option in enumerate(all_options):
        letter_mapping[letters[i]] = option
    
    # Build the formatted options string
    options_str = " ".join([f"({letters[i]}) {option}" for i, option in enumerate(all_options)])
    
    noise = torch.randn(160000)  # 10 seconds of noise at 16kHz
    # write to a temporary file
    audio_temp_path = f"{audio_dir}/{question_id}_temp_1.wav"
    sf.write(audio_temp_path, noise.numpy(), 16000)
    audio_path = audio_temp_path
    
    # Format the prompt for Qwen Audio - new format
    formatted_query = [
        {'audio': audio_path},
        {'text': f"Provide your best guess of this question. Think step by step.\n\nQuestion: {question}\n\nOptions: {options_str}\n\nWrite out your guess in the format of [[Letter]], such as [[A]], [[B]], [[C]], [[D]] or [[E]]."}
    ]
    
    return formatted_query, letter_mapping

def format_leave_one_out_prompt(question: str, question_id: str, audio_dir: str,
                               correct_answer: str, distractors: List[str], 
                               excluded_index: int) -> Tuple[List, Dict[str, str]]:
    """Format the prompt for leave-one-out evaluation with Qwen Audio."""
    # Create a new list without the excluded distractor
    available_distractors = distractors.copy()
    if excluded_index < len(distractors):
        available_distractors.pop(excluded_index)
    
    # Combine correct answer with remaining distractors
    all_options = [correct_answer] + available_distractors
    random.shuffle(all_options)
    
    # Create mapping of letters to options for later evaluation
    letter_mapping = {}
    letters = ["A", "B", "C", "D"][:len(all_options)]
    for i, option in enumerate(all_options):
        letter_mapping[letters[i]] = option
    
    # Build the formatted options string
    options_str = " ".join([f"({letters[i]}) {option}" for i, option in enumerate(all_options)])
    
    noise = torch.randn(160000)  # 10 seconds of noise at 16kHz
    # write to a temporary file
    audio_temp_path = f"{audio_dir}/{question_id}_temp_2.wav"
    sf.write(audio_temp_path, noise.numpy(), 16000)
    audio_path = audio_temp_path
    
    # Format the prompt for Qwen Audio using the new format structure
    formatted_query = [
        {'audio': audio_path},
        {'text': f"Provide your best guess of this question. Think step by step.\n\nQuestion: {question}\n\nOptions: {options_str}\n\nWrite out your guess in the format of [[Letter]], such as [[A]], [[B]], [[C]], [[D]]."}
    ]
    
    return formatted_query, letter_mapping

def extract_answer(response: str, letter_mapping: dict) -> str:
    """Extract the answer letter from the model's response.
    
    Args:
        response: The full text response from the model
        letter_mapping: Dictionary mapping answer letters to their text content
        
    Returns:
        The extracted answer letter or empty string if not found
    """
    # First try the specific format requested in the prompt
    pattern = r"\[\[([A-E])\]\]"
    matches = re.findall(pattern, response)
    
    if matches:
        return matches[-1]  # Return the last match
    
    # Try to find any letter mentions as a fallback
    letter_pattern = r"(?:answer is|chose|choose|select|guess is|I choose|I select|I guess) ([A-E])\b"
    letter_matches = re.findall(letter_pattern, response, re.IGNORECASE)
    if letter_matches:
        return letter_matches[-1].upper()
    
    # Look for bare letter mentions
    for letter in ["A", "B", "C", "D", "E"]:
        if f"option {letter}" in response or f"Option {letter}" in response or f"({letter})" in response:
            return letter
    
    # NEW: Check if any answer text is mentioned verbatim
    # Convert response and all options to lowercase for case-insensitive matching
    response_lower = response.lower()
    
    for letter, text in letter_mapping.items():
        # Skip if the text is empty or None
        if not text:
            continue
            
        # Check if the option text appears verbatim in the response
        if text.lower() in response_lower:
            return letter
    
    # If no clear answer found, return empty string
    return ""

# Update the evaluate_one_shot function to pass letter_mapping to extract_answer
def evaluate_one_shot(model, tokenizer, model_name: str, questions: Dict[str, Any], 
                     audio_dir: str, output_dir: Path, save_frequency: int = 5):
    """Run the one-shot evaluation with Qwen Audio."""
    existing_results = load_existing_outputs(output_dir, model_name, "one_shot")
    results = []
    
    items_processed = 0
    
    with tqdm(total=len(questions), desc=f"One-shot evaluation with {model_name}") as pbar:
        for question_id, question_data in questions.items():
            # Skip if already processed
            if question_id in existing_results:
                results.append(existing_results[question_id])
                pbar.update(1)
                continue
            
            formatted_query, letter_mapping = format_one_shot_prompt(
                question_data["question"],
                question_id,
                audio_dir,
                question_data["correct_answer"],
                question_data["distractors"]
            )
            
            response = get_model_response(model, tokenizer, formatted_query)
            # Pass letter_mapping to extract_answer
            answer_letter = extract_answer(response, letter_mapping)
            answer_text = letter_mapping.get(answer_letter, "")
            
            is_correct = answer_text == question_data["correct_answer"]
            
            result = {
                "question_id": question_id,
                "model_output": response,
                "extracted_answer_letter": answer_letter,
                "extracted_answer_text": answer_text,
                "correct_answer": question_data["correct_answer"],
                "is_correct": is_correct,
                "letter_mapping": letter_mapping
            }
            
            results.append(result)
            items_processed += 1
            pbar.update(1)
            
            # Save checkpoint periodically
            if items_processed % save_frequency == 0:
                save_checkpoint(results, output_dir, model_name, "one_shot")
                print(f"\nOne-shot checkpoint saved after processing {items_processed} items")
    
    # Final save
    save_checkpoint(results, output_dir, model_name, "one_shot")
    print(f"\nOne-shot evaluation complete. Results saved to {output_dir}/{model_name.replace('/', '_')}_one_shot.json")
    
    # Calculate and return accuracy
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    accuracy = correct_count / len(results) if results else 0
    return accuracy, results

# Update the evaluate_leave_one_out function to pass letter_mapping to extract_answer
def evaluate_leave_one_out(model, tokenizer, model_name: str, questions: Dict[str, Any], 
                          audio_dir: str, output_dir: Path, save_frequency: int = 5):
    """Run the leave-one-out evaluation with Qwen Audio."""
    existing_results = load_existing_outputs(output_dir, model_name, "leave_one_out")
    results = []
    
    items_processed = 0
    
    with tqdm(total=len(questions), desc=f"Leave-one-out evaluation with {model_name}") as pbar:
        for question_id, question_data in questions.items():
            # Skip if already processed
            if question_id in existing_results:
                results.append(existing_results[question_id])
                pbar.update(1)
                continue
            
            # For each question, run passes with different distractors removed
            passes = []
            extracted_answers = []
            
            # Run each pass (leave one distractor out)
            for excluded_idx in range(len(question_data["distractors"])):
                formatted_query, letter_mapping = format_leave_one_out_prompt(
                    question_data["question"],
                    question_id,
                    audio_dir,
                    question_data["correct_answer"],
                    question_data["distractors"],
                    excluded_idx
                )
                
                response = get_model_response(model, tokenizer, formatted_query)
                # Pass letter_mapping to extract_answer
                answer_letter = extract_answer(response, letter_mapping)
                answer_text = letter_mapping.get(answer_letter, "")
                
                is_correct = answer_text == question_data["correct_answer"]
                
                pass_result = {
                    "pass_number": excluded_idx,
                    "excluded_distractor": question_data["distractors"][excluded_idx] if excluded_idx < len(question_data["distractors"]) else None,
                    "model_output": response,
                    "extracted_answer_letter": answer_letter,
                    "extracted_answer_text": answer_text,
                    "is_correct": is_correct,
                    "letter_mapping": letter_mapping
                }
                
                passes.append(pass_result)
                if answer_text:  # Only count non-empty answers
                    extracted_answers.append(answer_text)
            
            # Determine the most agreed-upon answer
            most_common = Counter(extracted_answers).most_common(1)
            agreed_answer = most_common[0][0] if most_common else ""
            is_correct = agreed_answer == question_data["correct_answer"]
            
            result = {
                "question_id": question_id,
                "passes": passes,
                "most_agreed_answer": agreed_answer,
                "correct_answer": question_data["correct_answer"],
                "is_correct": is_correct,
                "vote_counts": dict(Counter(extracted_answers))
            }
            
            results.append(result)
            items_processed += 1
            pbar.update(1)
            
            # Save checkpoint periodically
            if items_processed % save_frequency == 0:
                save_checkpoint(results, output_dir, model_name, "leave_one_out")
                print(f"\nLeave-one-out checkpoint saved after processing {items_processed} items")
    
    # Final save
    save_checkpoint(results, output_dir, model_name, "leave_one_out")
    print(f"\nLeave-one-out evaluation complete. Results saved to {output_dir}/{model_name.replace('/', '_')}_leave_one_out.json")
    
    # Calculate and return accuracy
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    accuracy = correct_count / len(results) if results else 0
    return accuracy, results

def generate_summary(model_name: str, one_shot_results: List[Dict], leave_one_out_results: List[Dict],
                    output_dir: Path):
    """Generate a summary of the evaluation results."""
    one_shot_correct = sum(1 for r in one_shot_results if r.get("is_correct", False))
    one_shot_accuracy = one_shot_correct / len(one_shot_results) if one_shot_results else 0
    
    leave_one_out_correct = sum(1 for r in leave_one_out_results if r.get("is_correct", False))
    leave_one_out_accuracy = leave_one_out_correct / len(leave_one_out_results) if leave_one_out_results else 0
    
    # Calculate pass-level accuracy for leave-one-out
    pass_results = []
    for question in leave_one_out_results:
        for pass_data in question.get("passes", []):
            pass_results.append(pass_data.get("is_correct", False))
    
    pass_accuracy = sum(1 for p in pass_results if p) / len(pass_results) if pass_results else 0
    
    summary = {
        "model_name": model_name,
        "one_shot_accuracy": one_shot_accuracy,
        "one_shot_correct": one_shot_correct,
        "one_shot_total": len(one_shot_results),
        "leave_one_out_accuracy": leave_one_out_accuracy,
        "leave_one_out_correct": leave_one_out_correct,
        "leave_one_out_total": len(leave_one_out_results),
        "leave_one_out_pass_accuracy": pass_accuracy,
        "leave_one_out_pass_correct": sum(1 for p in pass_results if p),
        "leave_one_out_pass_total": len(pass_results)
    }
    
    # Save summary
    summary_path = output_dir / f"{model_name.replace('/', '_')}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation summary:")
    print(f"Model: {model_name}")
    print(f"One-shot accuracy: {one_shot_accuracy:.4f} ({one_shot_correct}/{len(one_shot_results)})")
    print(f"Leave-one-out accuracy: {leave_one_out_accuracy:.4f} ({leave_one_out_correct}/{len(leave_one_out_results)})")
    print(f"Leave-one-out pass-level accuracy: {pass_accuracy:.4f}")
    print(f"Summary saved to {summary_path}")
    
    return summary

def main():
    """Main function to run the evaluations with Qwen Audio."""
    parser = argparse.ArgumentParser(description="Evaluate Qwen Audio model on music questions")
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct", help="Qwen Audio model name")
    parser.add_argument("--input", default="/home/yongyi/muchomusic/filter.json", help="Path to input JSON file with questions")
    parser.add_argument("--audio-dir", required=True, help="Directory with audio files")
    parser.add_argument("--output-dir", default="evaluation_results_noise", help="Directory to save results")
    parser.add_argument("--save-frequency", type=int, default=5, 
                      help="Save checkpoint after processing this many items")
    parser.add_argument("--eval-types", default="both", choices=["one_shot", "leave_one_out", "both"],
                      help="Types of evaluation to run")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], 
                      help="Device to run the model on")
    # parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "auto"],
    #                   help="Precision for model inference")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(1234)
    
    # Load processor instead of tokenizer
    print(f"Loading processor from {args.model}...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    
    # Load model with appropriate precision
    print(f"Loading model {args.model} on {args.device}...")
    
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True
    }
    
    # if args.precision == "bf16":
    #     model_kwargs["bf16"] = True
    # elif args.precision == "fp16":
    #     model_kwargs["fp16"] = True
    
    from transformers import Qwen2AudioForConditionalGeneration
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model, **model_kwargs).eval()
    
    
    
    # Load questions
    questions = load_json_data(args.input)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    one_shot_results = []
    leave_one_out_results = []
    
    # Run evaluations based on selected types
    if args.eval_types in ["one_shot", "both"]:
        print(f"\nRunning one-shot evaluation with {args.model}...")
        one_shot_accuracy, one_shot_results = evaluate_one_shot(
            model, processor, args.model, questions, args.audio_dir, output_dir, args.save_frequency
        )
    
    if args.eval_types in ["leave_one_out", "both"]:
        print(f"\nRunning leave-one-out evaluation with {args.model}...")
        leave_one_out_accuracy, leave_one_out_results = evaluate_leave_one_out(
            model, processor, args.model, questions, args.audio_dir, output_dir, args.save_frequency
        )
    
    # Generate summary
    if args.eval_types == "both":
        generate_summary(args.model, one_shot_results, leave_one_out_results, output_dir)

if __name__ == "__main__":
    main()