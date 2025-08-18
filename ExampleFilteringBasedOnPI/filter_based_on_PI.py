import json
import math
import random
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from tqdm import tqdm
import numpy as np

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def create_prompt(question, caption=None, options=None):
    """
    Creates a prompt for a multiple-choice question.
    
    Args:
        question: The question text.
        caption: Optional caption/description.
        options: A dictionary of {letter: option_text}.
        
    Returns:
        A formatted prompt string.
    """
    options_text = "\n".join([f"{letter}. {option}" for letter, option in options.items()])
    prompt = f"Question: {question}\n"
    if caption:
        prompt += f"Context: {caption}\n"
    prompt += f"Options:\n{options_text}\nAnswer:"
    return prompt

def evaluate_all_options(llm, question, caption=None, options=None):
    """
    Calculates logprobs for each option for a given question in a single model call.
    
    Args:
        llm: The vLLM language model.
        question: The question text.
        caption: Optional caption/description.
        options: A list of options or a dict of {letter: option_text}.
        
    Returns:
        A dictionary mapping each option letter to its text and log probability.
    """
    if not options:
        return {}

    # Convert options to a dictionary if it's a list
    option_dict = options
    if isinstance(options, list):
        option_letters = [chr(65 + i) for i in range(len(options))] # A, B, C...
        option_dict = {letter: option for letter, option in zip(option_letters, options)}
    
    # The list of choice tokens we want to get probabilities for
    choice_tokens = list(option_dict.keys())
    
    # Create the base prompt that stops right before the answer
    prompt = create_prompt(question, caption, option_dict)
    
    # Configure sampling parameters to guide the decoding
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=len(choice_tokens), # Request logprobs for all choices
        guided_decoding=GuidedDecodingParams(choice=choice_tokens)
    )
    
    # Generate probabilities for all choices in a single pass
    outputs = llm.generate([prompt], sampling_params)
    
    # Extract the logprobs from the output
    prompt_logprobs = outputs[0].outputs[0].logprobs[0]
    
    results = {}
    for token_id, logprob_obj in prompt_logprobs.items():
        # The decoded token is the option letter (e.g., 'A', 'B')
        letter = logprob_obj.decoded_token.strip()
        if letter in option_dict:
            results[letter] = {
                "option": option_dict[letter],
                "logprob": logprob_obj.logprob
            }
            
    return results

def select_best_distractors(llm, question_data, num_final_distractors=4):
    """
    Selects the best distractors by evaluating them with the efficient function.
    (This function remains largely the same but now calls the refactored evaluate_all_options)
    """
    question = question_data["question"]
    caption = question_data["caption"]
    correct_answer = question_data["correct_answer"]
    distractors = question_data["distractors"]
    
    random.shuffle(distractors)
    
    if len(distractors) <= num_final_distractors:
        return distractors
        
    # Process in chunks to select promising candidates
    chunk_size = 6
    best_distractors = []
    
    for i in range(0, len(distractors), chunk_size):
        chunk = distractors[i:i+chunk_size]
        all_options = chunk + [correct_answer]
        
        results = evaluate_all_options(llm, question, caption, all_options)
        
        # Map letter back to the option text to identify which ones are distractors
        letter_to_option = {letter: res["option"] for letter, res in results.items()}
        
        sorted_results = sorted(
            [(letter, res["logprob"]) for letter, res in results.items() if letter_to_option[letter] != correct_answer],
            key=lambda x: x[1],
            reverse=True
        )
        
        best_from_chunk = [letter_to_option[letter] for letter, _ in sorted_results[:2]]
        best_distractors.extend(best_from_chunk)
        
    # Final evaluation round with the top candidates
    if len(best_distractors) > num_final_distractors:
        final_options = best_distractors + [correct_answer]
        final_results = evaluate_all_options(llm, question, caption, final_options)
        
        letter_to_option = {letter: res["option"] for letter, res in final_results.items()}
        
        sorted_final = sorted(
            [(letter, res["logprob"]) for letter, res in final_results.items() if letter_to_option[letter] != correct_answer],
            key=lambda x: x[1],
            reverse=True
        )
        
        best_distractors = [letter_to_option[letter] for letter, _ in sorted_final[:num_final_distractors]]
        
    return best_distractors

def assess_question_hardness(llm, question, caption, correct_answer, distractors, num_leave_one_out_groups=4):
    """
    Assesses the hardness of a question using leave-one-out probing.
    (Corrected to properly normalize log probabilities)
    """
    if len(distractors) != 4:
        raise ValueError(f"Expected exactly 4 distractors, got {len(distractors)}")

    all_probs = []
    
    for _ in range(num_leave_one_out_groups):
        leave_one_out_probs = []
        
        for i in range(4): # Create 4 groups, each leaving one distractor out
            selected_distractors = distractors[:i] + distractors[i+1:]
            current_options = selected_distractors + [correct_answer]
            random.shuffle(current_options)
            
            # Evaluate all options to get their log probabilities
            results = evaluate_all_options(llm, question, caption, current_options)
            
            # Extract logprobs and find the one for the correct answer
            logprobs = {res["option"]: res["logprob"] for res in results.values()}
            correct_logprob = logprobs[correct_answer]
            
            # Convert all logprobs to probabilities to correctly normalize
            # p = e^log(p)
            all_raw_probs = [math.exp(lp) for lp in logprobs.values()]
            sum_of_probs = sum(all_raw_probs)
            
            # The normalized probability of the correct answer
            if sum_of_probs > 0:
                normalized_correct_prob = math.exp(correct_logprob) / sum_of_probs
                leave_one_out_probs.append(normalized_correct_prob)
        
        if leave_one_out_probs:
            all_probs.append(sum(leave_one_out_probs) / len(leave_one_out_probs))
            
    avg_correct_prob = sum(all_probs) / len(all_probs) if all_probs else 0
    
    # Hardness is 1 - probability of being correct
    hardness = 1 - avg_correct_prob
    return hardness

def main():
    with open("distractors.json", "r") as f:
        data = json.load(f)
        
    # Initialize the model
    llm = LLM(model="Qwen/Qwen2.5-7B")
    
    filtered_data = {}
    question_hardness = {}
    
    for question_id, question_data in tqdm(data.items(), desc="Processing questions"):
        best_distractors = select_best_distractors(llm, question_data)
        
        filtered_data[question_id] = {
            "question": question_data["question"],
            "caption": question_data["caption"],
            "correct_answer": question_data["correct_answer"],
            "distractors": best_distractors
        }
        
        if len(best_distractors) == 4:
            hardness = assess_question_hardness(
                llm,
                question_data["question"],
                question_data["caption"],
                question_data["correct_answer"],
                best_distractors
            )
            filtered_data[question_id]["PI"] = hardness
            question_hardness[question_id] = hardness
            
    # Sort the data based on hardness score (descending)
    sorted_question_ids = sorted(question_hardness.keys(), key=lambda qid: question_hardness[qid], reverse=True)
    
    # Create a new sorted dictionary
    sorted_filtered_data = {qid: filtered_data[qid] for qid in sorted_question_ids}
    
    # Add any remaining questions (those without hardness scores) at the end
    for qid in filtered_data:
        if qid not in sorted_filtered_data:
            sorted_filtered_data[qid] = filtered_data[qid]
            
    with open("out.json", "w") as f:
        json.dump(sorted_filtered_data, f, indent=2)
        
    print(f"\nProcessed {len(data)} questions")
    print("Results saved to out.json, sorted by hardness score (hardest first)")

if __name__ == "__main__":
    main()
