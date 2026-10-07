import math
import torch

from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
from transformers import GPT2Tokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "/localstorage/home/f20220930/EasyEdit/huggingface_cache/gpt2-xl"

# Fact being edited
SUBJECT = "Peoria"

OLD_OBJECT = "Henri de Tonti"
NEW_OBJECT = "Jacques Marquette"

# ROME edit prompt
EDIT_PROMPT = "Peoria was established by"


# ============================================================
# QUERY SET
#
# We separate:
#
#   FORWARD queries:
#       subject -> object
#
#   REVERSE queries:
#       object -> subject
#
# This is important because the answer space is different
# in the two directions.
#
# The query construction follows the direct/reverse and
# refinement principles used in WikiProfile.
# ============================================================


FORWARD_QUERIES = [

    {
        "name": "direct",
        "type": "direct",

        "prompt":
            "What is the name of the explorer who established Peoria in 1691?",

        "old_answer": OLD_OBJECT,
        "new_answer": NEW_OBJECT,

        "specific": True,
        "minimal": True,
        "fact_grounded": True,
        "unique": True,
    },

    {
        "name": "direct_natural",
        "type": "direct_natural",

        "prompt":
            "Who was the explorer that founded Peoria in 1691?",

        "old_answer": OLD_OBJECT,
        "new_answer": NEW_OBJECT,

        "specific": True,
        "minimal": True,
        "fact_grounded": True,
        "unique": True,
    },

    {
        "name": "direct_completion",
        "type": "completion",

        "prompt":
            "The explorer who established Peoria in 1691 was",

        "old_answer": OLD_OBJECT,
        "new_answer": NEW_OBJECT,

        "specific": True,
        "minimal": True,
        "fact_grounded": True,
        "unique": True,
    },

    {
        "name": "direct_paraphrase",
        "type": "paraphrase",

        "prompt":
            "Peoria was established in 1691 by",

        "old_answer": OLD_OBJECT,
        "new_answer": NEW_OBJECT,

        "specific": True,
        "minimal": True,
        "fact_grounded": True,
        "unique": True,
    },
]


REVERSE_QUERIES = [

    {
        "name": "reverse",
        "type": "reverse",

        # OLD relationship:
        # Henri de Tonti -> Peoria
        "prompt":
            "What city did Henri de Tonti establish in 1691?",

        "expected_answer":
            "Peoria",

        "direction":
            "old",
        
        "specific": True,
        "minimal": True,
        "fact_grounded": True,
        "unique": True,
    },

    {
        "name": "reverse_natural",
        "type": "reverse_natural",

        "prompt":
            "What city did Henri de Tonti found in 1691?",

        "expected_answer":
            "Peoria",

        "direction":
            "old",

        "specific": True,
        "minimal": True,
        "fact_grounded": True,
        "unique": True,
    },

    {
        "name": "reverse_after_edit",
        "type": "reverse",

        # NEW relationship:
        # Jacques Marquette -> Peoria
        "prompt":
            "What city did Jacques Marquette establish in 1691?",

        "expected_answer":
            "Peoria",

        "direction":
            "new",

        "specific": True,
        "minimal": True,
        "fact_grounded": True,
        "unique": True,
    },

    {
        "name": "reverse_after_edit_natural",
        "type": "reverse_natural",

        "prompt":
            "What city did Jacques Marquette found in 1691?",

        "expected_answer":
            "Peoria",

        "direction":
            "new",

        "specific": True,
        "minimal": True,
        "fact_grounded": True,
        "unique": True,
    },
]


# ============================================================
# QUERY VALIDATION
# ============================================================

def validate_query(query):

    checks = {
        "specific": query["specific"],
        "minimal": query["minimal"],
        "fact_grounded": query["fact_grounded"],
        "unique": query["unique"],
    }

    valid = all(checks.values())

    return valid, checks


def validate_query_group(name, queries):

    print()
    print("=" * 75)
    print(f"{name.upper()} QUERY VALIDATION")
    print("=" * 75)

    valid_queries = []

    for query in queries:

        valid, checks = validate_query(query)

        print()
        print(f"[{query['name']}]")
        print(f"  Type   : {query['type']}")
        print(f"  Prompt : {query['prompt']}")

        print()
        print("  Validation:")

        for criterion, passed in checks.items():

            status = "PASS" if passed else "FAIL"

            print(
                f"    {criterion:<15}: {status}"
            )

        if valid:

            print("  RESULT : ACCEPTED")

            valid_queries.append(query)

        else:

            print("  RESULT : REJECTED")

    print()
    print(
        f"Accepted: {len(valid_queries)} / "
        f"{len(queries)}"
    )

    return valid_queries


# ============================================================
# TOKEN / PROBABILITY CALCULATION
# ============================================================

def get_token_info(
    model,
    tokenizer,
    prompt,
    answer,
    device
):

    # GPT-2 uses a leading space for normal word tokens.
    answer_text = " " + answer

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].to(device)

    answer_ids = tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].to(device)

    input_ids = torch.cat(
        [prompt_ids, answer_ids],
        dim=1
    )

    with torch.no_grad():

        outputs = model(input_ids)

        logits = outputs.logits

    prompt_len = prompt_ids.shape[1]

    answer_len = answer_ids.shape[1]

    token_log_probs = []

    for i in range(answer_len):

        token_position = prompt_len + i

        logits_at_position = logits[
            0,
            token_position - 1
        ]

        log_probs = torch.log_softmax(
            logits_at_position,
            dim=-1
        )

        token_id = answer_ids[0, i]

        token_log_prob = log_probs[token_id]

        token_log_probs.append(
            token_log_prob
        )

    token_log_probs = torch.stack(
        token_log_probs
    )

    # Length-normalized probability.
    mean_log_prob = (
        token_log_probs.mean().item()
    )

    probability = math.exp(
        mean_log_prob
    )

    # --------------------------------------------------------
    # FIRST TOKEN
    # --------------------------------------------------------

    first_logits = logits[
        0,
        prompt_len - 1
    ]

    first_probs = torch.softmax(
        first_logits,
        dim=-1
    )

    first_token_id = answer_ids[0, 0]

    first_token_probability = (
        first_probs[first_token_id].item()
    )

    sorted_indices = torch.argsort(
        first_probs,
        descending=True
    )

    first_token_rank = (
        (
            sorted_indices == first_token_id
        )
        .nonzero(as_tuple=True)[0]
        .item()
        + 1
    )

    return {
        "probability":
            probability,

        "first_token_probability":
            first_token_probability,

        "first_token_rank":
            first_token_rank,

        "num_tokens":
            answer_len,
    }


# ============================================================
# FORWARD QUERY EVALUATION
# ============================================================

def evaluate_forward_query(
    model,
    tokenizer,
    query,
    device
):

    old_info = get_token_info(
        model,
        tokenizer,
        query["prompt"],
        query["old_answer"],
        device
    )

    new_info = get_token_info(
        model,
        tokenizer,
        query["prompt"],
        query["new_answer"],
        device
    )

    old_log_prob = math.log(
        old_info["probability"]
    )

    new_log_prob = math.log(
        new_info["probability"]
    )

    margin = (
        new_log_prob
        - old_log_prob
    )

    return {
        "old": old_info,
        "new": new_info,
        "margin": margin,
    }


# ============================================================
# REVERSE QUERY EVALUATION
# ============================================================

def evaluate_reverse_query(
    model,
    tokenizer,
    query,
    device
):

    answer_info = get_token_info(
        model,
        tokenizer,
        query["prompt"],
        query["expected_answer"],
        device
    )

    return {
        "answer": answer_info
    }


# ============================================================
# GENERATE FORWARD PROFILE
# ============================================================

def generate_forward_profile(
    model,
    tokenizer,
    queries,
    device
):

    results = []

    for query in queries:

        result = evaluate_forward_query(
            model,
            tokenizer,
            query,
            device
        )

        results.append({
            "query": query,
            "result": result,
        })

    return results


# ============================================================
# GENERATE REVERSE PROFILE
# ============================================================

def generate_reverse_profile(
    model,
    tokenizer,
    queries,
    device
):

    results = []

    for query in queries:

        result = evaluate_reverse_query(
            model,
            tokenizer,
            query,
            device
        )

        results.append({
            "query": query,
            "result": result,
        })

    return results


# ============================================================
# PRINT FORWARD PROFILE
# ============================================================

def print_forward_profile(
    title,
    results
):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)

    for item in results:

        query = item["query"]

        result = item["result"]

        old = result["old"]
        new = result["new"]

        print()
        print(query["name"])

        print(
            f"  Prompt: {query['prompt']}"
        )

        print(
            f"  OLD {OLD_OBJECT:<25} "
            f"P={old['probability']:.6f}, "
            f"first-P="
            f"{old['first_token_probability']:.6f}, "
            f"rank={old['first_token_rank']}, "
            f"tokens={old['num_tokens']}"
        )

        print(
            f"  NEW {NEW_OBJECT:<25} "
            f"P={new['probability']:.6f}, "
            f"first-P="
            f"{new['first_token_probability']:.6f}, "
            f"rank={new['first_token_rank']}, "
            f"tokens={new['num_tokens']}"
        )

        print(
            f"  NEW-vs-OLD margin = "
            f"{result['margin']:.6f}"
        )


# ============================================================
# PRINT REVERSE PROFILE
# ============================================================

def print_reverse_profile(
    title,
    results
):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)

    for item in results:

        query = item["query"]

        answer = item["result"]["answer"]

        print()
        print(query["name"])

        print(
            f"  Direction: "
            f"{query['direction']}"
        )

        print(
            f"  Prompt: "
            f"{query['prompt']}"
        )

        print(
            f"  Expected answer: "
            f"{query['expected_answer']}"
        )

        print(
            f"  P={answer['probability']:.6f}, "
            f"first-P="
            f"{answer['first_token_probability']:.6f}, "
            f"rank={answer['first_token_rank']}, "
            f"tokens={answer['num_tokens']}"
        )


# ============================================================
# PRINT FORWARD CHANGES
# ============================================================

def print_forward_changes(
    pre_results,
    post_results
):

    print()
    print("=" * 75)
    print("FORWARD CHANGES AFTER EDIT")
    print("=" * 75)

    for pre_item, post_item in zip(
        pre_results,
        post_results
    ):

        query = pre_item["query"]

        pre = pre_item["result"]

        post = post_item["result"]

        new_delta = (
            post["new"]["probability"]
            - pre["new"]["probability"]
        )

        old_delta = (
            post["old"]["probability"]
            - pre["old"]["probability"]
        )

        margin_delta = (
            post["margin"]
            - pre["margin"]
        )

        print()
        print(query["name"])

        print(
            f"  Δ NEW probability: "
            f"{new_delta:+.6f}"
        )

        print(
            f"  Δ OLD probability: "
            f"{old_delta:+.6f}"
        )

        print(
            f"  Δ margin: "
            f"{margin_delta:+.6f}"
        )

        print(
            f"  NEW rank: "
            f"{pre['new']['first_token_rank']}"
            f" -> "
            f"{post['new']['first_token_rank']}"
        )

        print(
            f"  OLD rank: "
            f"{pre['old']['first_token_rank']}"
            f" -> "
            f"{post['old']['first_token_rank']}"
        )


# ============================================================
# PRINT REVERSE CHANGES
# ============================================================

def print_reverse_changes(
    pre_results,
    post_results
):

    print()
    print("=" * 75)
    print("REVERSE CHANGES AFTER EDIT")
    print("=" * 75)

    for pre_item, post_item in zip(
        pre_results,
        post_results
    ):

        query = pre_item["query"]

        pre = pre_item["result"]["answer"]

        post = post_item["result"]["answer"]

        probability_delta = (
            post["probability"]
            - pre["probability"]
        )

        print()
        print(query["name"])

        print(
            f"  Direction: "
            f"{query['direction']}"
        )

        print(
            f"  Δ Peoria probability: "
            f"{probability_delta:+.6f}"
        )

        print(
            f"  Rank: "
            f"{pre['first_token_rank']}"
            f" -> "
            f"{post['first_token_rank']}"
        )


# ============================================================
# PARAMETER CHANGE
# ============================================================

def get_rome_weight(model):

    return (
        model
        .transformer
        .h[17]
        .mlp
        .c_proj
        .weight
    )


def compare_weights(
    before_weight,
    after_weight
):

    delta = (
        after_weight.float()
        - before_weight.float()
    ).abs()

    print()
    print("=" * 75)
    print("PARAMETER CHANGE")
    print("=" * 75)

    print(
        f"Max parameter change : "
        f"{delta.max().item():.10f}"
    )

    print(
        f"Mean parameter change: "
        f"{delta.mean().item():.10f}"
    )

    print(
        f"Nonzero parameters  : "
        f"{(delta > 0).sum().item()}"
    )

    print(
        f"Total parameters     : "
        f"{delta.numel()}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    pre_forward,
    post_forward,
    pre_reverse,
    post_reverse
):

    print()
    print("=" * 75)
    print("EXPERIMENT SUMMARY")
    print("=" * 75)

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    pre_forward_margins = [
        item["result"]["margin"]
        for item in pre_forward
    ]

    post_forward_margins = [
        item["result"]["margin"]
        for item in post_forward
    ]

    post_new_probs = [
        item["result"]["new"]["probability"]
        for item in post_forward
    ]

    post_old_probs = [
        item["result"]["old"]["probability"]
        for item in post_forward
    ]

    margin_changes = [
        post - pre
        for pre, post in zip(
            pre_forward_margins,
            post_forward_margins
        )
    ]

    print()
    print("FORWARD")

    print(
        f"  Mean NEW probability: "
        f"{sum(post_new_probs) / len(post_new_probs):.6f}"
    )

    print(
        f"  Mean OLD probability: "
        f"{sum(post_old_probs) / len(post_old_probs):.6f}"
    )

    print(
        f"  Mean NEW-vs-OLD margin: "
        f"{sum(post_forward_margins) / len(post_forward_margins):.6f}"
    )

    print(
        f"  Minimum NEW-vs-OLD margin: "
        f"{min(post_forward_margins):.6f}"
    )

    print(
        f"  Mean margin improvement: "
        f"{sum(margin_changes) / len(margin_changes):.6f}"
    )

    forward_robust = all(
        margin > 0
        for margin in post_forward_margins
    )

    print(
        f"  NEW preferred on every "
        f"forward query: "
        f"{forward_robust}"
    )

    # --------------------------------------------------------
    # REVERSE
    # --------------------------------------------------------

    print()
    print("REVERSE")

    for pre_item, post_item in zip(
        pre_reverse,
        post_reverse
    ):

        query = post_item["query"]

        pre_prob = (
            pre_item["result"]
            ["answer"]["probability"]
        )

        post_prob = (
            post_item["result"]
            ["answer"]["probability"]
        )

        pre_rank = (
            pre_item["result"]
            ["answer"]["first_token_rank"]
        )

        post_rank = (
            post_item["result"]
            ["answer"]["first_token_rank"]
        )

        print()
        print(
            f"  {query['name']}"
        )

        print(
            f"    Direction: "
            f"{query['direction']}"
        )

        print(
            f"    Probability: "
            f"{pre_prob:.6f}"
            f" -> "
            f"{post_prob:.6f}"
        )

        print(
            f"    Rank: "
            f"{pre_rank}"
            f" -> "
            f"{post_rank}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # VALIDATE QUERIES
    # ========================================================

    valid_forward = validate_query_group(
        "Forward",
        FORWARD_QUERIES
    )

    valid_reverse = validate_query_group(
        "Reverse",
        REVERSE_QUERIES
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print()
    print("=" * 75)
    print("LOADING TOKENIZER")
    print("=" * 75)

    tokenizer = GPT2Tokenizer.from_pretrained(
        MODEL_PATH
    )

    # ========================================================
    # EASYEDIT
    # ========================================================

    print()
    print("=" * 75)
    print("LOADING EASYEDIT")
    print("=" * 75)

    hparams = ROMEHyperParams.from_hparams(
        "./hparams/ROME/gpt2-xl"
    )

    editor = BaseEditor.from_hparams(
        hparams
    )

    model = editor.model

    device = next(
        model.parameters()
    ).device

    print(
        f"Model device: {device}"
    )

    # ========================================================
    # TOKENIZATION CHECK
    # ========================================================

    print()
    print("=" * 75)
    print("TOKENIZATION CHECK")
    print("=" * 75)

    for answer in [
        OLD_OBJECT,
        NEW_OBJECT,
        SUBJECT
    ]:

        ids = tokenizer(
            " " + answer,
            add_special_tokens=False
        )["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(
            ids
        )

        print()
        print(answer)
        print(f"  IDs    : {ids}")
        print(f"  Tokens : {tokens}")

    # ========================================================
    # PRE-EDIT FORWARD
    # ========================================================

    pre_forward = generate_forward_profile(
        model,
        tokenizer,
        valid_forward,
        device
    )

    print_forward_profile(
        "PRE-EDIT FORWARD PROFILE",
        pre_forward
    )

    # ========================================================
    # PRE-EDIT REVERSE
    # ========================================================

    pre_reverse = generate_reverse_profile(
        model,
        tokenizer,
        valid_reverse,
        device
    )

    print_reverse_profile(
        "PRE-EDIT REVERSE PROFILE",
        pre_reverse
    )

    # ========================================================
    # SAVE ORIGINAL ROME WEIGHT
    # ========================================================

    before_weight = (
        get_rome_weight(model)
        .detach()
        .clone()
    )

    # ========================================================
    # APPLY ROME
    # ========================================================

    print()
    print("=" * 75)
    print("APPLYING ROME")
    print("=" * 75)

    metrics, edited_model, _ = editor.edit(

        prompts=[EDIT_PROMPT],

        ground_truth=[OLD_OBJECT],

        target_new=[NEW_OBJECT],

        subject=[SUBJECT],

        # Keep the edited weights in memory.
        sequential_edit=True,
    )

    print()
    print("EASYEDIT METRICS")
    print(metrics)

    # ========================================================
    # PARAMETER CHANGE
    # ========================================================

    after_weight = (
        get_rome_weight(edited_model)
        .detach()
        .clone()
    )

    compare_weights(
        before_weight,
        after_weight
    )

    # ========================================================
    # POST-EDIT FORWARD
    # ========================================================

    post_forward = generate_forward_profile(
        edited_model,
        tokenizer,
        valid_forward,
        device
    )

    print_forward_profile(
        "POST-EDIT FORWARD PROFILE",
        post_forward
    )

    # ========================================================
    # POST-EDIT REVERSE
    # ========================================================

    post_reverse = generate_reverse_profile(
        edited_model,
        tokenizer,
        valid_reverse,
        device
    )

    print_reverse_profile(
        "POST-EDIT REVERSE PROFILE",
        post_reverse
    )

    # ========================================================
    # CHANGES
    # ========================================================

    print_forward_changes(
        pre_forward,
        post_forward
    )

    print_reverse_changes(
        pre_reverse,
        post_reverse
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(
        pre_forward,
        post_forward,
        pre_reverse,
        post_reverse
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 75)
    print("EXPERIMENT FINISHED")
    print("=" * 75)


if __name__ == "__main__":
    main()
