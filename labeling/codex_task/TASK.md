# Annotation task for an independent third annotator (Codex)

You are annotator 3 for a research dataset: 1,118 Reddit texts about fresh produce (avocados, lettuce, onions,
cucumbers). Two other annotators (both Claude models) have labelled the same texts. A third annotator from a
different model family lets us measure how much the labels depend on one model family. **Your labels must be
independent**, so:

- Read only `labeling/codebook.md` (the label table and the "Decision rules"; ignore its "Procedure" section) and the
  input files in this folder (`input_01.jsonl` … `input_12.jsonl`).
- Do **not** open `labeling/label_sample.csv`, `labeling/labels.csv`, `labeling/reviews*.json`,
  `labeling/annotations/`, `labeling/review_page.html`, `outputs/`, `data/processed/classified_observations*`,
  or any other file that contains labels, predictions or annotations.
- Judge every text yourself by reading it. Do **not** write a keyword rule, script, heuristic or classifier, and do
  not call any other model or API to label the texts. Scripts are fine only for checking your output files.

## What to label

Each input line is one JSON object: `obs_id`, `product`, `record_type` (post or comment), `subreddit`, `text`.
Give **one** label for the market of **that row's `product`** (a text naming two products appears once per product
and may get a different label for each):

| Label | Use it when the text… |
|---|---|
| price | talks about the product's price, a price change, affordability, a discount with a price |
| supply | talks about availability, shortages, empty shelves, poor quality from supply problems, crop or weather conditions, imports, recalls or the product being pulled |
| demand | is mainly about buying or eating more or less of the product: recipes centred on it, what to do with a lot of it, stocking up, trends, a promotion without a price (e.g. free guacamole), avoiding or throwing it out after a scare |
| other | no clear market signal: the product is one ingredient among several or mentioned in passing, cooking or storage tips, home gardening, jokes, restaurant meal complaints, product-flavoured snacks |

Key rules (full list in the codebook): price > supply > demand when a text has two signals; home-garden problems are
other unless they describe crop conditions in general; when in doubt between a signal and other, choose other.

## Output

For each input file `input_NN.jsonl` write `output_NN.jsonl` in this folder, one line per input line, same order:

```
{"obs_id": "...", "label": "price|supply|demand|other", "reason": "<= 15 words on why"}
```

Work file by file. When finished, check that every output file has as many lines as its input file, every obs_id
appears once, and every label is one of the four values, then commit the 12 output files (nothing else).
