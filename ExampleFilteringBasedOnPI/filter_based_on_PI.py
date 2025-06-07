import json
import math
import random
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def get_last_token_logprobs(llm, sampling_params, prompt):
    """Calculate the probability of the next token for a given prompt."""
    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0]
    key, value = next(iter(output.prompt_logprobs[-1].items()))
    return math.exp(value.logprob) 
    # Please let me know if there's a better way to handle this
    # I tried so hard and I thought there has to be a better way to do this than running the model once for each option
    # Why doesn't VLLM support passing in a set of tokens to get logprobs for all of them at once?
    # Or it does and I just don't know how to do it?
    # Please, please email Yongyi Zang at zyy0116@gmail.com if you know how to do this better

def create_prompt(question, caption=None, options=None, format_type="default"):
    """
    Create a flexible prompt for the LLM with various formatting options.
    
    Args:
        question: The question text
        caption: Optional caption/description (default: None)
        options: Dict of {letter: option_text} or list of options (default: None)
        format_type: The prompt format style (default: "default", can be "minimal", "verbose", etc.)
        
    Returns:
        Prompt string 
    """
    options_text = "; ".join([f"{letter}. {option}" for letter, option in options.items()])
    prompt = f"Provide your best guess of this question. "
    prompt += f"The question is: {question} "
    prompt += f"The answer candidates are: {options_text}. Answer without the parenthesis."
    
    return prompt

def evaluate_all_options(llm, sampling_params, question, caption=None, options=None, format_type="default"):
    """
    Calculate logprobs for each option for a given question.
    
    Args:
        llm: The language model
        sampling_params: Sampling parameters for the language model
        question: The question text
        caption: Optional caption/description (default: None)
        options: List of options or dict of {letter: option_text} (default: None)
        format_type: The prompt format style (default: "default")
        
    Returns:
        Dictionary mapping options to their logprobs
    """
    if options is None:
        return {}
    
    # Convert options to dictionary if it's a list
    option_dict = options
    if isinstance(options, list):
        # Generate letters dynamically based on number of options
        option_letters = [chr(65 + i) for i in range(len(options))]  # A, B, C, D, E, etc.
        option_dict = {letter: option for letter, option in zip(option_letters, options)}
    
    results = {}
    
    # Evaluate each option as the "correct" answer
    for letter, option in option_dict.items():
        base_prompt = create_prompt(question, caption, option_dict, format_type)
        prompt = f"{base_prompt} The most likely answer is {letter}"
        logprob = get_last_token_logprobs(llm, sampling_params, prompt)
        results[letter] = {
            "option": option,
            "logprob": logprob
        }
    
    return results

def select_best_distractors(llm, sampling_params, question_data, num_final_distractors=4):
    """
    Select the best distractors by evaluating them with the new flexible functions.
    
    Args:
        llm: The language model
        sampling_params: Sampling parameters for the language model
        question_data: Dictionary containing question, caption, correct_answer, and distractors
        num_final_distractors: Number of final distractors to return
        
    Returns:
        List of the best distractors
    """
    question = question_data["question"]
    caption = question_data["caption"]
    correct_answer = question_data["correct_answer"]
    distractors = question_data["distractors"]
    
    # randomize the distractors
    random.shuffle(distractors)
    
    # If we already have the right number of distractors, return them
    if len(distractors) <= num_final_distractors:
        return distractors
    
    # Process all distractors in chunks to make the process more efficient
    chunk_size = min(6, len(distractors))  # Process up to 6 distractors at a time
    best_distractors = []
    
    for i in range(0, len(distractors), chunk_size):
        chunk = distractors[i:i+chunk_size]
        
        # Create options with the current chunk and the correct answer
        all_options = chunk + [correct_answer]
        
        # Get logprobs for all options
        results = evaluate_all_options(llm, sampling_params, question, caption, all_options)
        
        # Sort the distractors by logprob in descending order (higher logprob = more challenging)
        # Map letter back to index in the all_options list
        letter_to_index = {chr(65 + j): j for j in range(len(all_options))}
        
        sorted_results = sorted(
            [(letter, res["logprob"]) for letter, res in results.items() 
             if all_options[letter_to_index[letter]] != correct_answer],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Add the best distractors from this chunk
        best_from_chunk = [all_options[letter_to_index[letter]] for letter, _ in sorted_results[:min(len(sorted_results), 2)]]
        best_distractors.extend(best_from_chunk)
    
    # If we have more than the required number, take the top ones
    if len(best_distractors) > num_final_distractors:
        # Get logprobs for all of the best distractors together
        final_options = best_distractors + [correct_answer]
        final_results = evaluate_all_options(llm, sampling_params, question, caption, final_options)
        
        # Sort by logprob and take the top num_final_distractors
        letter_to_index = {chr(65 + j): j for j in range(len(final_options))}
        
        sorted_final = sorted(
            [(letter, res["logprob"]) for letter, res in final_results.items() 
             if final_options[letter_to_index[letter]] != correct_answer],
            key=lambda x: x[1],
            reverse=True
        )
        
        best_distractors = [final_options[letter_to_index[letter]] for letter, _ in sorted_final[:num_final_distractors]]
    
    return best_distractors

def assess_question_hardness(llm, sampling_params, question, caption, correct_answer, distractors, num_leave_one_out_groups=4):
    """
    Assess the hardness of a question using leave-one-out probing.
    
    Args:
        llm: The language model
        sampling_params: Sampling parameters for the language model
        question: The question text
        caption: Optional caption/description
        correct_answer: The correct answer text
        distractors: List of distractor texts
        num_leave_one_out_groups: Number of leave-one-out groups to evaluate
        
    Returns:
        Hardness score (1 - probability of answering correctly)
    """
    # Ensure we have exactly 4 distractors
    if len(distractors) != 4:
        raise ValueError(f"Expected exactly 4 distractors, got {len(distractors)}")
    
    # Combine correct answer with distractors for full option set
    all_options = distractors + [correct_answer]
    
    correct_probs = []
    
    # Do leave-one-out probing num_leave_one_out_groups times
    for _ in range(num_leave_one_out_groups):
        leave_one_out_probs = []
        
        # Create 4 groups, each leaving out one distractor
        for i in range(4):
            # Select 3 distractors (leaving one out) and the correct answer
            selected_distractors = distractors.copy()
            left_out = selected_distractors.pop(i)
            current_options = selected_distractors + [correct_answer]
            
            # Randomize the order
            random.shuffle(current_options)
            
            # Evaluate all options to get their probabilities
            results = evaluate_all_options(llm, sampling_params, question, caption, current_options)
            
            # Extract logprobs
            logprobs = [res["logprob"] for letter, res in results.items()]
            sum_logprobs = sum(logprobs)
            
            # Find the normalized probability for the correct answer
            correct_index = current_options.index(correct_answer)
            correct_letter = chr(65 + correct_index)
            correct_prob = results[correct_letter]["logprob"] / sum_logprobs
            
            leave_one_out_probs.append(correct_prob)
        
        # Average the probabilities from the 4 leave-one-out groups
        correct_probs.append(sum(leave_one_out_probs) / len(leave_one_out_probs))
    
    # Final probability is the average across all num_leave_one_out_groups runs
    avg_correct_prob = sum(correct_probs) / len(correct_probs)
    
    # Hardness is 1 - probability of answering correctly
    hardness = 1 - avg_correct_prob
    
    return hardness

def main():
    # Load distractors.json
    with open("distractors.json", "r") as f:
        data = json.load(f)
    
    # Initialize the model
    model_name = "Qwen/Qwen2.5-7B"
    llm = LLM(model=model_name)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, prompt_logprobs=1)
    
    # Output dictionary
    filtered_data = {}
    question_hardness = {}
    
    # Process each question with progress bar
    for question_id, question_data in tqdm(data.items(), desc="Processing questions"):
        best_distractors = select_best_distractors(llm, sampling_params, question_data)
        
        # Save the processed data
        filtered_data[question_id] = {
            "question": question_data["question"],
            "caption": question_data["caption"],
            "correct_answer": question_data["correct_answer"],
            "distractors": best_distractors
        }
        
        # Assess question hardness if we have exactly 4 distractors
        if len(best_distractors) == 4:
            hardness = assess_question_hardness(
                llm,
                sampling_params,
                question_data["question"],
                question_data["caption"],
                question_data["correct_answer"],
                best_distractors
            )
            
            # Store hardness in the question data
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
        
        # Save the filtered and sorted data
        with open("out.json", "w") as f:
            json.dump(sorted_filtered_data, f, indent=2)
    
    print(f"Processed {len(data)} questions")
    print(f"Questions are sorted by hardness score (hardest first)")

if __name__ == "__main__":
    main()