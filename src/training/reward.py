from utils.extraction import extract_info, normalize_ward_name, translate_ward_jp_to_en


def match_format_exactly(completions, **kwargs):
  scores = []
  for completion in completions:
    score = 0.
    response = completion[0]["content"]
    answer_json = extract_info(response)
    if answer_json["observation"] is not None:
      score += 1.
    if answer_json["reasoning"] is not None:
      score += 1.
    if answer_json["answer"] is not None:
      score += 1.
    scores.append(score)
  return scores


def check_answer(completions, answer, **kwargs):
    """
    Evaluates the predicted answer against the ground truth.

    Args:
        completions (list): List of model generated completions.
        answer (dict): The ground truth answer dictionary with 'ward' and 'town'.

    Returns:
        float: The score based on the correctness of the ward prediction.
    """
    scores = []
    for i, (completion, answer) in enumerate(zip(completions, answer)):
        score = -4.5  # Default score for no answer

        response = completion[0]["content"]
        extracted_data = extract_info(response)
        predicted_answer = extracted_data.get("answer", {})
        predicted_ward = predicted_answer.get("ward")
        reasoning = extracted_data.get("reasoning", "") # Get reasoning for partial match

        # Normalize the predicted ward name
        normalized_predicted_ward = normalize_ward_name(predicted_ward) if predicted_ward else None

        # Translate and normalize the ground truth ward name
        ground_truth_ward = answer.get("ward")
        translated_normalized_ground_truth_ward = translate_ward_jp_to_en(ground_truth_ward) if ground_truth_ward else None

        if normalized_predicted_ward and translated_normalized_ground_truth_ward:
            # Check for exact match (after normalization and translation)
            if normalized_predicted_ward == translated_normalized_ground_truth_ward:
                score = 5.0
            # Check if normalized predicted ward is mentioned in the reasoning
            elif reasoning and normalized_predicted_ward in normalize_ward_name(reasoning).lower():
                 score = 2.0
            else:
                score = -2.5 # Incorrect answer

        scores.append(score)
    return scores