import json
import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F
import tqdm
from findfile import find_file, find_cwd_dir
from termcolor import colored
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModel

from pyabsa.framework.flag_class.flag_template import (
    LabelPaddingOption,
    TaskCodeOption,
    DeviceTypeOption,
)
from pyabsa.framework.prediction_class.predictor_template import InferenceModel
from pyabsa.utils.data_utils.dataset_item import DatasetItem
from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset
from pyabsa.utils.pyabsa_utils import set_device, print_args, fprint
from ..dataset_utils.__lcf__.atepc_utils import (
    load_atepc_inference_datasets,
    process_iob_tags,
)
from ..dataset_utils.__lcf__.data_utils_for_inference import (
    ATEPCProcessor,
    convert_ate_examples_to_features,
    convert_apc_examples_to_features,
)
from ..dataset_utils.__lcf__.data_utils_for_training import split_aspect
from ..models import ATEPCModelList

# ----- Sentence splitting (wtpsplit SaT with safe fallback) -----
try:
    from wtpsplit import SaT
    _HAS_SAT = True
except Exception:
    _HAS_SAT = False

MERGE_CONJ = {"and", "&", ","}
HARD_CONNECTORS = {"but", "however", "yet", "although", "though", ";", ","}

def _merge_coordinated_aspect_snippets(item_dict, pol_snips, include_connector=True):
    """
    Merge snippets for coordinated aspects with the same sentiment, e.g.:
      'The camera and lens are excellent' -> one positive snippet
    instead of two ('The camera', 'lens are excellent').

    item_dict fields used: tokens, aspect, sentiment, position
    pol_snips: {"positive": [...], "negative": [...]}
    """
    tokens = item_dict.get("tokens") or []
    aspects = item_dict.get("aspect") or []
    sentiments = item_dict.get("sentiment") or []
    positions = item_dict.get("position") or []

    if not tokens or not aspects or not sentiments or not positions:
        return pol_snips

    # Build normalized spans for each aspect
    spans = []
    for i, pos in enumerate(positions):
        if not pos:
            continue
        if isinstance(pos, list):
            start = min(pos)
            end = max(pos)
        else:
            start = end = int(pos)
        spans.append({
            "idx": i,
            "start": start,
            "end": end,
            "sent": (sentiments[i] or "").lower(),
        })

    # Sort spans by start
    spans.sort(key=lambda x: x["start"])

    merged_ranges = []   # list of (start_idx, end_idx, sentiment)
    i = 0
    n = len(spans)
    while i < n:
        j = i
        cur_sent = spans[i]["sent"]
        left = spans[i]["start"]
        right = spans[i]["end"]

        # attempt to expand group to the right while:
        #  - same sentiment
        #  - between spans we only see a coordinator like 'and' / '&' / ',' (no 'but', 'however', etc.)
        #  - no hard contrast connector appears between
        while j + 1 < n and spans[j + 1]["sent"] == cur_sent:
            gap_l = spans[j]["end"]
            gap_r = spans[j + 1]["start"]
            between = [t.lower() for t in tokens[gap_l:gap_r+1]]

            # stop if any hard connector sits in between (contrast boundary)
            if any(w in {"but", "however", "yet", ";"} for w in between):
                break
            # must have a coordinating token like 'and', '&', or a bare comma
            if not any(w in MERGE_CONJ for w in between):
                break

            # ok, we can merge with the next span
            j += 1
            right = spans[j]["end"]

        # if a group larger than 1 span was formed, create one merged range
        if j > i:
            merged_ranges.append((left, right, cur_sent))
            i = j + 1
        else:
            i += 1

    if not merged_ranges:
        return pol_snips  # nothing to merge

    # Build merged snippets, extending to the predicate and stopping at punctuation/contrast
    def _extend_right_to_predicate(r):
        R = r
        # stop at first hard connector or comma/semicolon that likely ends the predicate
        for k in range(r, len(tokens)):
            w = tokens[k].lower()
            if w in {"but", "however", "yet"} or w in {",", ";"}:
                return k - 1 if not include_connector else k
        return len(tokens) - 1

    def _extend_left_article(l):
        L = l
        # include preceding determiner/article if present
        if L - 1 >= 0 and tokens[L-1].lower() in {"the", "a", "an", "this", "that", "these", "those"}:
            L = L - 1
        return max(0, L)

    pos_merged, neg_merged = set(pol_snips.get("positive", [])), set(pol_snips.get("negative", []))

    for L, R, sent in merged_ranges:
        L2 = _extend_left_article(L)
        R2 = _extend_right_to_predicate(R)
        snippet = " ".join(tokens[L2:R2+1]).strip()

        if sent.startswith("pos"):
            # remove sub-snippets that are contained in the merged one
            to_remove = [s for s in pos_merged if s and s in snippet]
            for s in to_remove:
                pos_merged.discard(s)
            pos_merged.add(snippet)
        elif sent.startswith("neg"):
            to_remove = [s for s in neg_merged if s and s in snippet]
            for s in to_remove:
                neg_merged.discard(s)
            neg_merged.add(snippet)

    return {"positive": list(pos_merged), "negative": list(neg_merged)}

_SPLITTER = None
def _get_splitter():
    global _SPLITTER
    if _SPLITTER is not None:
        return _SPLITTER
    if _HAS_SAT:
        try:
            sat = SaT("sat-3l-sm")
            try:
                sat = sat.half()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    sat = sat.to("cuda")
            except Exception:
                pass
            _SPLITTER = ("sat", sat)
            return _SPLITTER
        except Exception:
            pass
    # Fallback: simple connector/punctuation-based split
    CONNECTORS = ["; however ,", "however ,", "; however,", "however,", " but ", " yet ", ";", ".", " and "]
    def _fallback_split(text: str):
        tmp = text
        for c in CONNECTORS:
            tmp = tmp.replace(c, "|")
        parts = [cl.strip() for cl in tmp.split("|")]
        return [cl for cl in parts if cl]
    _SPLITTER = ("fallback", _fallback_split)
    return _SPLITTER

def _split_into_clauses(text: str):
    kind, sp = _get_splitter()
    if kind == "sat":
        return sp.split([text])[0]
    else:
        return sp(text)
# ---------------------------------------------------------------


# ----------------------- NEW: helper for aspect-centric windows -----------------------
def _aspect_windows_by_polarity(item, window=5, include_connector=True):
    """
    Build aspect-centric snippets grouped by polarity using token indices.

    Priority rules:
      - LEFT boundary prefers STRONG connectors (however, but, yet, although, though, nevertheless, nonetheless, whereas, while, still)
        over punctuation/comma. If only a comma exists, skip it (avoid starting with ',').
      - RIGHT boundary stops at nearest punctuation/comma/connector/end.
      - Fallback: +/- `window` tokens.
      - Light detokenization for clean punctuation spacing.
    """
    tokens = item.get("tokens", [])
    aspects = item.get("aspect", [])
    positions = item.get("position", [])
    sentiments = item.get("sentiment", [])

    if not tokens or not aspects or not positions or not sentiments:
        return {"positive": [], "negative": [], "neutral": []}

    # Sets
    PUNCT = {".", "!", "?", ";", ":"}
    STRONG_CONNS = {"but", "however", "though", "although", "yet", "nevertheless", "nonetheless", "whereas", "while", "still"}
    SOFT_CONNS   = {"and", "or", "nor", "so", "because"}
    CONNS = STRONG_CONNS | SOFT_CONNS

    # Helpers
    def is_comma(tok: str) -> bool:
        return tok == ","

    def is_punct(tok: str) -> bool:
        return tok in PUNCT

    def is_conn(tok: str) -> bool:
        return tok.lower() in CONNS

    def is_strong_conn(tok: str) -> bool:
        return tok.lower() in STRONG_CONNS

    def detok(ts):
        s = " ".join(ts)
        for p in [" .", " ,", " !", " ?", " ;", " :"]:
            s = s.replace(p, p.strip())
        # also normalize 'however ,'
        s = s.replace("however ,", "however,")
        return " ".join(s.split())

    # scan LEFT with priority: nearest strong connector > nearest punctuation > nearest comma > nearest soft connector
    def left_boundary_index(start_idx: int) -> int:
        idx_strong = idx_punct = idx_comma = idx_soft = -1
        i = start_idx
        while i >= 0:
            t = tokens[i]
            if idx_strong < 0 and is_strong_conn(t):
                idx_strong = i
                # we can break early if we want the nearest strong connector only
                break
            if idx_punct < 0 and is_punct(t):
                idx_punct = i
            if idx_comma < 0 and is_comma(t):
                idx_comma = i
            if idx_soft < 0 and (is_conn(t) and not is_strong_conn(t)):
                idx_soft = i
            i -= 1
        # priority choose:
        if idx_strong >= 0:
            return idx_strong
        if idx_punct >= 0:
            return idx_punct
        if idx_comma >= 0:
            return idx_comma
        if idx_soft >= 0:
            return idx_soft
        return -1  # no boundary; before sentence

    # scan RIGHT: nearest punctuation or connector or comma; else end
    def right_boundary_index(start_idx: int) -> int:
        n = len(tokens)
        for i in range(start_idx, n):
            t = tokens[i]
            if is_punct(t) or is_comma(t) or is_conn(t):
                return i
        return n  # after sentence

    def append_unique(lst, s, seen):
        s_norm = " ".join(s.split()).strip()
        if s_norm and s_norm not in seen:
            lst.append(s_norm)
            seen.add(s_norm)

    pos_snips, neg_snips, neu_snips = [], [], []
    seen_pos, seen_neg, seen_neu = set(), set(), set()

    for i, pos in enumerate(positions):
        if i >= len(sentiments):
            continue

        # token span of aspect (inclusive)
        if isinstance(pos, (list, tuple)):
            l_tok = pos[0]
            r_tok = pos[-1] if len(pos) > 1 else pos[0]
        else:
            l_tok = r_tok = int(pos)

        # boundaries
        lb = left_boundary_index(l_tok - 1)
        rb = right_boundary_index(r_tok + 1)

        # default clip between boundaries (excluding boundary tokens)
        L = lb + 1
        R = rb - 1

        # fallback if invalid
        if L > R or L < 0 or R >= len(tokens):
            L = max(0, l_tok - window)
            R = min(len(tokens) - 1, r_tok + window)
            lb = -1  # so connector logic below won't pull in leftovers

        # include connector on the left if it's a strong one
        if include_connector and lb >= 0:
            if is_strong_conn(tokens[lb]):
                L = lb  # include 'however' / 'but' etc.
            elif is_punct(tokens[lb]):
                # punctuation left is okay to include for context like '; however ,'
                # but if immediately after is 'however', prefer to include that, not raw punct
                if lb + 1 < len(tokens) and is_strong_conn(tokens[lb + 1]):
                    L = lb + 1
                else:
                    # usually we keep punctuation excluded
                    L = lb + 1
            elif is_comma(tokens[lb]):
                # don't start with a comma
                L = lb + 1
            else:
                # soft connector ('and', 'because', etc.): keep it excluded by default
                L = max(L, lb + 1)

        # trim leading commas in the final range
        while L <= R and is_comma(tokens[L]):
            L += 1

        snippet = detok(tokens[L:R + 1])

        pol = str(sentiments[i]).strip().lower()
        if pol == "positive":
            append_unique(pos_snips, snippet, seen_pos)
        elif pol == "negative":
            append_unique(neg_snips, snippet, seen_neg)
        else:
            append_unique(neu_snips, snippet, seen_neu)

    return {"positive": pos_snips, "negative": neg_snips, "neutral": neu_snips}
# --------------------------------------------------------------------------------------


class AspectExtractor(InferenceModel):
    """Predictor for Aspect Term Extraction and (optional) Polarity Classification.

    Loads an ATEPC checkpoint and provides utilities to extract aspect terms
    from text, with optional sentiment classification for extracted aspects
    depending on the configured model/task. Supports single-text and batch
    inference and can auto-detect dataset files for inference.
    """

    task_code = TaskCodeOption.Aspect_Term_Extraction_and_Classification

    def __init__(self, checkpoint=None, **kwargs):
        """Initialize the ATEPC aspect extractor from a trained checkpoint.

        Args:
            checkpoint: Path to a checkpoint directory or a tuple returned
                by the trainer (model, config, tokenizer).
            **kwargs: Optional keyword arguments such as `auto_device`,
                `offline`, and `verbose`.

        Raises:
            RuntimeError: If the checkpoint cannot be loaded.
            ValueError: If an unsupported fine-tuned checkpoint is provided.
        """
        # load from a trainer
        super().__init__(checkpoint, task_code=self.task_code, **kwargs)

        if self.checkpoint and not isinstance(self.checkpoint, str):
            fprint("Load aspect extractor from trainer")
            self.model = self.checkpoint[0]
            self.config = self.checkpoint[1]
            self.tokenizer = self.checkpoint[2]
        else:
            if "fine-tuned" in self.checkpoint:
                raise ValueError(
                    "Do not support to directly load a fine-tuned model, please load a .state_dict or .model instead!"
                )
            fprint("Load aspect extractor from", self.checkpoint)
            try:
                state_dict_path = find_file(
                    self.checkpoint, ".state_dict", exclude_key=["__MACOSX"]
                )
                model_path = find_file(
                    self.checkpoint, ".model", exclude_key=["__MACOSX"]
                )
                tokenizer_path = find_file(
                    self.checkpoint, ".tokenizer", exclude_key=["__MACOSX"]
                )
                config_path = find_file(
                    self.checkpoint, ".config", exclude_key=["__MACOSX"]
                )

                fprint("config: {}".format(config_path))
                fprint("state_dict: {}".format(state_dict_path))
                fprint("model: {}".format(model_path))
                fprint("tokenizer: {}".format(tokenizer_path))

                with open(config_path, mode="rb") as f:
                    self.config = pickle.load(f)
                    self.config.auto_device = kwargs.get("auto_device", True)
                    set_device(self.config, self.config.auto_device)

                if state_dict_path or model_path:
                    if state_dict_path:
                        if kwargs.get("offline", False):
                            try:
                                self.bert = AutoModel.from_pretrained(
                                    find_cwd_dir(
                                        self.config.pretrained_bert.split("/")[-1]
                                    ),
                                    trust_remote_code=True,
                                )
                            except Exception:
                                self.bert = AutoModel.from_pretrained(
                                    find_cwd_dir(
                                        self.config.pretrained_bert.split("/")[-1]
                                    )
                                )
                        else:
                            try:
                                self.bert = AutoModel.from_pretrained(
                                    self.config.pretrained_bert,
                                    trust_remote_code=True,
                                )
                            except Exception:
                                self.bert = AutoModel.from_pretrained(
                                    self.config.pretrained_bert,
                                )

                        self.model = self.config.model(self.bert, self.config)
                        self.model.load_state_dict(
                            torch.load(
                                state_dict_path, map_location=DeviceTypeOption.CPU
                            ),
                            strict=False,
                        )
                    elif model_path:
                        self.model = torch.load(
                            model_path, map_location=DeviceTypeOption.CPU
                        )
                    with open(tokenizer_path, mode="rb") as f:
                        try:
                            if kwargs.get("offline", False):
                                try:
                                    self.tokenizer = AutoTokenizer.from_pretrained(
                                        find_cwd_dir(
                                            self.config.pretrained_bert.split("/")[-1]
                                        ),
                                        do_lower_case="uncased"
                                        in self.config.pretrained_bert,
                                        trust_remote_code=True,
                                    )
                                except Exception:
                                    self.tokenizer = AutoTokenizer.from_pretrained(
                                        find_cwd_dir(
                                            self.config.pretrained_bert.split("/")[-1]
                                        ),
                                        do_lower_case="uncased"
                                        in self.config.pretrained_bert,
                                        trust_remote_code=True,
                                        use_fast=False,
                                    )
                            else:
                                try:
                                    self.tokenizer = AutoTokenizer.from_pretrained(
                                        self.config.pretrained_bert,
                                        do_lower_case="uncased"
                                        in self.config.pretrained_bert,
                                        trust_remote_code=True,
                                    )
                                except Exception:
                                    self.tokenizer = AutoTokenizer.from_pretrained(
                                        self.config.pretrained_bert,
                                        do_lower_case="uncased"
                                        in self.config.pretrained_bert,
                                        trust_remote_code=True,
                                        use_fast=False,
                                    )
                        except ValueError:
                            self.tokenizer = pickle.load(f)

            except Exception as e:
                raise RuntimeError(
                    "Exception: {} Fail to load the model from {}! ".format(
                        e, self.checkpoint
                    )
                )

            if not hasattr(ATEPCModelList, self.model.__class__.__name__):
                raise KeyError(
                    "The checkpoint you are loading is not from any ATEPC model."
                )

        self.processor = ATEPCProcessor(self.tokenizer)
        self.num_labels = len(self.config.label_list) + 1

        # ----------------------- NEW: defaults for snippet config -----------------------
        if not hasattr(self.config, "context_radius"):
            # tokens to include on EACH side of aspect
            self.config.context_radius = 5
        if not hasattr(self.config, "include_connector_in_snippet"):
            # include one connector token (e.g., 'but') before the window
            self.config.include_connector_in_snippet = True
        # -------------------------------------------------------------------------------

        if kwargs.get("verbose", False):
            fprint("Config used in Training:")
            print_args(self.config)

        if self.config.gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                    self.config.gradient_accumulation_steps
                )
            )

        self.eval_dataloader = None

        self.__post_init__(**kwargs)

    def merge_result(self, sentence_res, results):
        """merge ate sentence result and apc results, and restore to original sentence order
        Args:
            sentence_res ([tuple]): list of ate sentence results, which has (tokens, iobs)
            results ([dict]): list of apc results
        Returns:
            [dict]: merged extraction/polarity results for each input example
        """
        final_res = []
        if results["polarity_res"] is not None:
            merged_results = OrderedDict()
            pre_example_id = None
            # merge ate and apc results, assume they are same ordered
            for item1, item2 in zip(results["extraction_res"], results["polarity_res"]):
                cur_example_id = item1[3]
                assert (
                    cur_example_id == item2["example_id"]
                ), "ate and apc results should be same ordered"
                if pre_example_id is None or cur_example_id != pre_example_id:
                    merged_results[cur_example_id] = {
                        "sentence": item2["sentence"],
                        "aspect": [item2["aspect"]],
                        "position": [item2["pos_ids"]],
                        "sentiment": [item2["sentiment"]],
                        "probs": [item2["probs"]],
                        "confidence": [item2["confidence"]],
                    }
                else:
                    merged_results[cur_example_id]["aspect"].append(item2["aspect"])
                    merged_results[cur_example_id]["position"].append(item2["pos_ids"])
                    merged_results[cur_example_id]["sentiment"].append(
                        item2["sentiment"]
                    )
                    merged_results[cur_example_id]["probs"].append(item2["probs"])
                    merged_results[cur_example_id]["confidence"].append(
                        item2["confidence"]
                    )
                # remember example id
                pre_example_id = item1[3]
            for i, item in enumerate(sentence_res):
                asp_res = merged_results.get(i)
                item_dict = {
                    "sentence": " ".join(item[0]),
                    "IOB": item[1],
                    "tokens": item[0],
                    "aspect": asp_res["aspect"] if asp_res else [],
                    "position": asp_res["position"] if asp_res else [],
                    "sentiment": asp_res["sentiment"] if asp_res else [],
                    "probs": asp_res["probs"] if asp_res else [],
                    "confidence": asp_res["confidence"] if asp_res else [],
                }

                # ----------------------- NEW: attach positive / negative snippets -----------------------
                pol_snips = _aspect_windows_by_polarity(
                    item_dict,
                    window=getattr(self.config, "context_radius", 5),
                    include_connector=getattr(self.config, "include_connector_in_snippet", True),
                )
                item_dict["positive"] = pol_snips.get("positive", [])
                item_dict["negative"] = pol_snips.get("negative", [])
                # If you also want neutral:
                # item_dict["neutral"] = pol_snips.get("neutral", [])
                # -----------------------------------------------------------------------------------------

                final_res.append(item_dict)
        else:
            for item1, item2 in zip(sentence_res, results["extraction_res"]):
                item_dict = {
                    "sentence": " ".join(item2[0]),
                    "IOB": item2[1],
                    "tokens": item1[0],
                    "aspect": item2[3],
                    "position": [],
                    "sentiment": [],
                    "probs": [],
                    "confidence": [],
                }

                # no sentiments in this branch → empty snippet buckets
                item_dict["positive"] = []
                item_dict["negative"] = []
                # item_dict["neutral"] = []

                final_res.append(item_dict)

        return final_res

    def extract_aspect(
        self,
        inference_source: Union[List[Path], list, str],
        save_result=True,
        print_result=True,
        pred_sentiment=True,
        **kwargs
    ):
        """
        Extract aspects and their corresponding polarities from a list of input files.

        Args:
            self: An instance of the model class.
            inference_source: A list of file paths, or a directory containing files to be processed.
            save_result (bool): Whether to save the output to a file. Default is True.
            print_result (bool): Whether to print the output to the console. Default is True.
            pred_sentiment (bool): Whether to predict the sentiment of each aspect. Default is True.
            **kwargs: Additional keyword arguments to be passed to the `batch_predict` method.

        Returns:
            The predicted aspects and their corresponding polarities.
        """
        return self.batch_predict(
            inference_source, save_result, print_result, pred_sentiment, **kwargs
        )

    def predict(
    self,
        text,
        save_result=True,
        print_result=True,
        pred_sentiment=True,
        **kwargs
    ):
        """
        Single string:
        - Split into clauses (wtpsplit SaT if available, else a simple fallback)
        - Run ATEPC per clause via batch_predict
        - Merge clause-level outputs back into one record

        List[str]:
        - Unchanged: delegate to batch_predict
        """

        # ---------- Local, no-global splitter helpers (no name collisions) ----------
        def _get_sat_local():
            try:
                from wtpsplit import SaT
            except Exception:
                return None
            try:
                sat = SaT("sat-3l-sm")
                try:
                    sat = sat.half()
                except Exception:
                    pass
                try:
                    import torch
                    if torch.cuda.is_available():
                        sat = sat.to("cuda")
                except Exception:
                    pass
                return sat
            except Exception:
                return None

        def _split_clauses_local(s: str):
            # try SaT first
            sat = _get_sat_local()
            if sat is not None:
                try:
                    # SaT API: split([text]) -> [list_of_sentences]
                    return sat.split([s])[0]
                except Exception:
                    pass
            # fallback: light rule-based split
            tmp = s
            for sep in ["; however,", "however,", " but ", " yet ", ";", ".", " and "]:
                tmp = tmp.replace(sep, "|")
            parts = [p.strip() for p in tmp.split("|")]
            return [p for p in parts if p]
        # ---------------------------------------------------------------------------

        # If a single string, split and merge
        if isinstance(text, str):
            # Clause segmentation (purely local, cannot collide with model attrs)
            clauses = _split_clauses_local(text)
            if not clauses:
                clauses = [text]

            # Run ATEPC per clause; rely on existing, working pipeline
            clause_results = self.batch_predict(
                clauses,
                save_result=False,
                print_result=False,
                pred_sentiment=pred_sentiment,
                **kwargs,
            )

            # Merge fields back into one record
            merged = {
                "sentence": text,
                "aspect": [],
                "position": [],
                "sentiment": [],
                "probs": [],
                "confidence": [],
                "tokens": [],
                "clauses": clause_results,  # keep per-clause outputs for debugging
            }

            for co in clause_results:
                if not isinstance(co, dict):
                    continue
                # capture first available tokens for convenience
                if not merged["tokens"] and co.get("tokens"):
                    merged["tokens"] = co.get("tokens", [])
                merged["aspect"].extend(co.get("aspect", []) or [])
                merged["position"].extend(co.get("position", []) or [])
                merged["sentiment"].extend(co.get("sentiment", []) or [])
                merged["probs"].extend(co.get("probs", []) or [])
                merged["confidence"].extend(co.get("confidence", []) or [])

            # Combine clause snippets if merge_result added them
            pos_all, neg_all = [], []
            for co in clause_results:
                if isinstance(co, dict):
                    pos_all.extend(co.get("positive", []) or [])
                    neg_all.extend(co.get("negative", []) or [])
            merged["positive"] = pos_all
            merged["negative"] = neg_all

            if print_result:
                import json as _json
                print(_json.dumps(merged, indent=2, ensure_ascii=False))
            if save_result:
                import json as _json
                with open("atepc_inference.json", "w", encoding="utf8") as f:
                    f.write(_json.dumps([merged], indent=2, ensure_ascii=False))
            return merged

        # If a list[str], keep original behavior
        return self.batch_predict(
            text,
            save_result=save_result,
            print_result=print_result,
            pred_sentiment=pred_sentiment,
            **kwargs,
        )


    def batch_predict(
        self,
        target_file: Union[List[Path], list, str],
        save_result=True,
        print_result=True,
        pred_sentiment=True,
        **kwargs
    ):
        """
        Args:
            target_file (list): list of input examples or a list of files to be predicted
            save_result (bool, optional): save result to file. Defaults to True.
            print_result (bool, optional): print result to console. Defaults to True.
            pred_sentiment (bool, optional): predict sentiment. Defaults to True.
        Returns:
        """

        self.config.eval_batch_size = kwargs.get("eval_batch_size", 32)

        results = {"extraction_res": None, "polarity_res": None}
        if isinstance(target_file, DatasetItem) or isinstance(target_file, str):
            # using integrated inference dataset
            inference_set = detect_infer_dataset(
                target_file, task_code=TaskCodeOption.Aspect_Polarity_Classification
            )
            target_file = load_atepc_inference_datasets(inference_set)

        elif isinstance(target_file, list):
            pass

        else:
            raise ValueError(
                "Please run inference using examples list or inference dataset path (list)!"
            )

        if target_file:
            extraction_res, sentence_res = self._extract(target_file)
            if not pred_sentiment:
                filtered_res = []
                for i, res in enumerate(extraction_res):
                    bio_tags = res[1]
                    aspect = []
                    for idx, tag in enumerate(bio_tags):
                        if "B-ASP" in tag:
                            aspect.append(res[0][idx])
                        elif "I-ASP" in tag and aspect:
                            aspect[-1] += " " + res[0][idx]
                    if not filtered_res:
                        filtered_res.append((res[0], aspect, res[2], aspect))
                    else:
                        if filtered_res[-1][0] != res[0]:
                            filtered_res.append((res[0], res[1], res[2], aspect))
                        else:
                            filtered_res[-1][1].extend(aspect)
                extraction_res = filtered_res
            results["extraction_res"] = extraction_res
            if pred_sentiment:
                results["polarity_res"] = self._run_prediction(
                    results["extraction_res"]
                )
            results = self.merge_result(sentence_res, results)
            if save_result:
                save_path = os.path.join(
                    os.getcwd(),
                    "{}.{}.result.json".format(
                        self.config.task_name, self.config.model.__name__
                    ),
                )
                fprint(
                    "The results of aspect term extraction have been saved in {}".format(
                        save_path
                    )
                )
                with open(save_path, "w", encoding="utf8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            if print_result:
                for ex_id, r in enumerate(results):
                    colored_text = r["sentence"][:]
                    for aspect, sentiment, confidence in zip(
                        r["aspect"], r["sentiment"], r["confidence"]
                    ):
                        if sentiment.upper() == "POSITIVE":
                            colored_aspect = colored(
                                "<{}:{} Confidence:{}>".format(
                                    aspect, sentiment, confidence
                                ),
                                "green",
                            )
                        elif sentiment.upper() == "NEUTRAL":
                            colored_aspect = colored(
                                "<{}:{} Confidence:{}>".format(
                                    aspect, sentiment, confidence
                                ),
                                "cyan",
                            )
                        elif sentiment.upper() == "NEGATIVE":
                            colored_aspect = colored(
                                "<{}:{} Confidence:{}>".format(
                                    aspect, sentiment, confidence
                                ),
                                "red",
                            )
                        else:
                            colored_aspect = colored(
                                "<{}:{} Confidence:{}>".format(
                                    aspect, sentiment, confidence
                                ),
                                "magenta",
                            )
                        colored_text = colored_text.replace(
                            " {} ".format(aspect), " {} ".format(colored_aspect), 1
                        )
                    res_format = "Example {}: {}".format(ex_id, colored_text)
                    fprint(res_format)

            return results

    # Temporal code, pending configimization
    def _extract(self, examples):
        sentence_res = []  # extraction result by sentence
        extraction_res = []  # extraction result flatten by aspect

        self.infer_dataloader = None
        examples = self.processor.get_examples_for_aspect_extraction(examples)
        infer_features = convert_ate_examples_to_features(
            examples,
            self.config.label_list,
            self.config.max_seq_len,
            self.tokenizer,
            self.config,
        )
        all_spc_input_ids = torch.tensor(
            [f.input_ids_spc for f in infer_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in infer_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in infer_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in infer_features], dtype=torch.long
        )
        all_polarities = torch.tensor(
            [f.polarity for f in infer_features], dtype=torch.long
        )
        all_valid_ids = torch.tensor(
            [f.valid_ids for f in infer_features], dtype=torch.long
        )
        all_lmask_ids = torch.tensor(
            [f.label_mask for f in infer_features], dtype=torch.long
        )

        all_tokens = [f.tokens for f in infer_features]
        infer_data = TensorDataset(
            all_spc_input_ids,
            all_segment_ids,
            all_input_mask,
            all_label_ids,
            all_polarities,
            all_valid_ids,
            all_lmask_ids,
        )
        # Run prediction for full raw_data
        infer_sampler = SequentialSampler(infer_data)
        self.infer_dataloader = DataLoader(
            infer_data,
            sampler=infer_sampler,
            pin_memory=True,
            batch_size=self.config.eval_batch_size,
        )

        # extract_aspects
        self.model.eval()
        if "index_to_IOB_label" not in self.config.args:
            label_map = {i: label for i, label in enumerate(self.config.label_list, 1)}
        else:
            label_map = self.config.index_to_IOB_label
        if len(infer_data) >= 100:
            it = tqdm.tqdm(self.infer_dataloader, desc="extracting aspect terms")
        else:
            it = self.infer_dataloader
        for i_batch, (
            input_ids_spc,
            segment_ids,
            input_mask,
            label_ids,
            polarity,
            valid_ids,
            l_mask,
        ) in enumerate(it):
            input_ids_spc = input_ids_spc.to(self.config.device)
            segment_ids = segment_ids.to(self.config.device)
            input_mask = input_mask.to(self.config.device)
            label_ids = label_ids.to(self.config.device)
            polarity = polarity.to(self.config.device)
            valid_ids = valid_ids.to(self.config.device)
            l_mask = l_mask.to(self.config.device)
            with torch.no_grad():
                ate_logits, apc_logits = self.model(
                    input_ids_spc,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                    polarity=polarity,
                    valid_ids=valid_ids,
                    attention_mask_label=l_mask,
                )
            if self.config.use_bert_spc:
                label_ids = self.model.get_batch_token_labels_bert_base_indices(
                    label_ids
                )
            ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
            ate_logits = ate_logits.detach().cpu().numpy()
            label_ids = label_ids.to(DeviceTypeOption.CPU).numpy()
            for i, i_ate_logits in enumerate(ate_logits):
                pred_iobs = []
                sentence_res.append(
                    (all_tokens[i + (self.config.eval_batch_size * i_batch)], pred_iobs)
                )
                for j, m in enumerate(label_ids[i]):
                    if j == 0:
                        continue
                    elif len(pred_iobs) == len(
                        all_tokens[i + (self.config.eval_batch_size * i_batch)]
                    ):
                        break
                    else:
                        pred_iobs.append(label_map.get(i_ate_logits[j], "O"))

                ate_result = []
                polarity = []
                for t, l in zip(
                    all_tokens[i + (self.config.eval_batch_size * i_batch)], pred_iobs
                ):
                    ate_result.append("{}({})".format(t, l))
                    if "ASP" in l:
                        polarity.append(
                            abs(LabelPaddingOption.SENTIMENT_PADDING)
                        )  # 1 tags the valid position aspect terms
                    else:
                        polarity.append(LabelPaddingOption.SENTIMENT_PADDING)

                POLARITY_PADDING = [LabelPaddingOption.SENTIMENT_PADDING] * len(
                    polarity
                )
                example_id = i_batch * self.config.eval_batch_size + i
                pred_iobs = process_iob_tags(pred_iobs)
                for idx in range(1, len(polarity)):
                    if polarity[idx - 1] != str(
                        LabelPaddingOption.SENTIMENT_PADDING
                    ) and split_aspect(pred_iobs[idx - 1], pred_iobs[idx]):
                        _polarity = polarity[:idx] + POLARITY_PADDING[idx:]
                        polarity = POLARITY_PADDING[:idx] + polarity[idx:]
                        extraction_res.append(
                            (
                                all_tokens[i + (self.config.eval_batch_size * i_batch)],
                                pred_iobs,
                                _polarity,
                                example_id,
                            )
                        )

                    if (
                        polarity[idx] != str(LabelPaddingOption.SENTIMENT_PADDING)
                        and idx == len(polarity) - 1
                        and split_aspect(pred_iobs[idx])
                    ):
                        _polarity = polarity[: idx + 1] + POLARITY_PADDING[idx + 1 :]
                        polarity = POLARITY_PADDING[: idx + 1] + polarity[idx + 1 :]
                        extraction_res.append(
                            (
                                all_tokens[i + (self.config.eval_batch_size * i_batch)],
                                pred_iobs,
                                _polarity,
                                example_id,
                            )
                        )

        return extraction_res, sentence_res

    def _run_prediction(self, examples):
        res = []  # sentiment classification result
        # ate example id map to apc example id
        example_id_map = dict([(apc_id, ex[3]) for apc_id, ex in enumerate(examples)])

        self.infer_dataloader = None
        examples = self.processor.get_examples_for_sentiment_classification(examples)
        infer_features = convert_apc_examples_to_features(
            examples,
            self.config.label_list,
            self.config.max_seq_len,
            self.tokenizer,
            self.config,
        )
        all_spc_input_ids = torch.tensor(
            [f.input_ids_spc for f in infer_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in infer_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in infer_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in infer_features], dtype=torch.long
        )
        all_valid_ids = torch.tensor(
            [f.valid_ids for f in infer_features], dtype=torch.long
        )
        all_lmask_ids = torch.tensor(
            [f.label_mask for f in infer_features], dtype=torch.long
        )
        lcf_cdm_vec = torch.tensor(
            [f.lcf_cdm_vec for f in infer_features], dtype=torch.float32
        )
        lcf_cdw_vec = torch.tensor(
            [f.lcf_cdw_vec for f in infer_features], dtype=torch.float32
        )
        all_tokens = [f.tokens for f in infer_features]
        all_aspects = [f.aspect for f in infer_features]
        all_positions = [f.positions for f in infer_features]
        infer_data = TensorDataset(
            all_spc_input_ids,
            all_segment_ids,
            all_input_mask,
            all_label_ids,
            all_valid_ids,
            all_lmask_ids,
            lcf_cdm_vec,
            lcf_cdw_vec,
        )
        # Run prediction for full raw_data
        self.model.config.use_bert_spc = True

        infer_sampler = SequentialSampler(infer_data)
        self.infer_dataloader = DataLoader(
            infer_data,
            sampler=infer_sampler,
            pin_memory=True,
            batch_size=self.config.eval_batch_size,
        )

        # extract_aspects
        self.model.eval()

        # Correct = {True: 'Correct', False: 'Wrong'}
        if len(infer_data) >= 100:
            it = tqdm.tqdm(self.infer_dataloader, desc="classifying aspect sentiments")
        else:
            it = self.infer_dataloader
        for i_batch, batch in enumerate(it):
            (
                input_ids_spc,
                segment_ids,
                input_mask,
                label_ids,
                valid_ids,
                l_mask,
                lcf_cdm_vec,
                lcf_cdw_vec,
            ) = batch
            input_ids_spc = input_ids_spc.to(self.config.device)
            segment_ids = segment_ids.to(self.config.device)
            input_mask = input_mask.to(self.config.device)
            label_ids = label_ids.to(self.config.device)
            valid_ids = valid_ids.to(self.config.device)
            l_mask = l_mask.to(self.config.device)
            lcf_cdm_vec = lcf_cdm_vec.to(self.config.device)
            lcf_cdw_vec = lcf_cdw_vec.to(self.config.device)
            with torch.no_grad():
                ate_logits, apc_logits = self.model(
                    input_ids_spc,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                    valid_ids=valid_ids,
                    attention_mask_label=l_mask,
                    lcf_cdm_vec=lcf_cdm_vec,
                    lcf_cdw_vec=lcf_cdw_vec,
                )
                for i, i_apc_logits in enumerate(apc_logits):
                    if (
                        "index_to_label" in self.config.args
                        and int(i_apc_logits.argmax(axis=-1))
                        in self.config.index_to_label
                    ):
                        sent = self.config.index_to_label.get(
                            int(i_apc_logits.argmax(axis=-1))
                        )
                    else:
                        sent = int(torch.argmax(i_apc_logits, -1))
                    result = {}
                    probs = [
                        float(x)
                        for x in F.softmax(i_apc_logits, dim=-1).cpu().numpy().tolist()
                    ]
                    apc_id = i_batch * self.config.eval_batch_size + i
                    result["sentence"] = " ".join(all_tokens[apc_id])
                    result["tokens"] = all_tokens[apc_id]
                    result["probs"] = probs
                    result["confidence"] = round(max(probs), 4)
                    result["aspect"] = all_aspects[apc_id]
                    result["pos_ids"] = [x - 1 for x in all_positions[apc_id]]
                    result["sentiment"] = sent
                    result["example_id"] = example_id_map[apc_id]
                    res.append(result)

        return res


class Predictor(AspectExtractor):
    pass