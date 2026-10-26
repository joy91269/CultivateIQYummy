# Labelling codebook: market signal of a post or comment

Each row of `label_sample.csv` is one text **about one product** (column `product`). A text that mentions two
products appears once per product and may get a different label for each. Pick the **one** label that best
describes what the text says about the market for *that* product.

| Label | Use it when the text… | Examples |
|---|---|---|
| **price** | talks about the product's price, a price change, affordability, sticker shock, a price comparison, or a discount *with* a price | "Avocados were $2.49 each at Kroger, insane." · "Lettuce prices doubled since summer." · "Cucumbers 3 for $1 this week." |
| **supply** | talks about availability, shortages, empty shelves, out-of-stock, poor quality caused by supply problems, crop or weather conditions, harvest, imports/tariffs/border closures, transport problems, recalls or the product being pulled from shelves | "No romaine anywhere, store says shortage." · "Freeze wiped out my onion starts." · "Mexico avocado imports suspended." · "McDonald's pulled onions from the Quarter Pounder." |
| **demand** | is mainly about wanting, buying, or eating more/less of the product: purchases, stocking up, cravings, recipes or diet trends *centred on the product*, promotions that drive buying, avoiding the product (e.g. after a food-safety scare) | "I've been buying cucumbers every day for that TikTok salad." · "Not touching onions until the E. coli thing is over." · "What do I do with 10 lbs of avocados?" |
| **other** | does not carry a clear price, supply or demand signal for the product: the product is just one ingredient in a recipe, a cooking technique question, gardening tips without a crop/market problem, jokes, off-topic | "Recipe: brown the beef, add onions, garlic, cumin…" · "How do you keep cut avocado from browning?" |

## Decision rules

1. **Centrality.** If the product is only one item in a list of ingredients or is mentioned in passing, label
   **other** — even if the text is a recipe. Recipes count as **demand** only when the product is the focus
   (e.g. a cucumber-salad post).
2. **Price beats supply beats demand.** If a text clearly contains two signals, choose in the order
   price > supply > demand. ("Avocados are $3 because of the import ban" → price.)
3. **Food safety.** Reports of a recall, outbreak or the product being pulled → **supply** (availability).
   People saying they will stop or avoid buying → **demand**.
4. **Gardening.** Home-garden yield problems (blight, pests, weather) → **supply** only when they describe crop
   conditions in general or a bad season; routine growing tips → **other**.
5. **Promotions.** A sale with a price → **price**; a promotion or menu campaign without a price → **demand**.
6. **Restaurant meals.** The price of a whole meal that merely contains the product (a $15 burrito bowl with
   lettuce) → **other**; the price of the product itself or of a product-specific add-on (the extra charge for
   guacamole) → **price**; a restaurant running out of the product → **supply**; complaints about portion
   sizes in general → **other**.
7. When in doubt between a signal and **other**, choose **other**. Use the `notes` column for hard cases.

## Procedure used in this project

1. `labeling/make_label_sample.py` drew a stratified sample of posts (product × keyword candidate), then
   keyword-enriched batches of posts and comments so that the rare price and supply classes are represented;
   one text per thread.
2. **Annotator 1** (Claude Opus, the AI assistant that built this pipeline) labelled every row following this
   codebook (`label_ai1`, with a short reason).
3. **Annotator 2** (Claude Sonnet, a different model, run in a separate session that saw only this codebook and the
   texts, never annotator 1's labels) labelled every row independently (`label_ai2`). Both annotators come from the
   same model family, so their errors may be correlated; this is why agreed labels are also spot-checked by a
   person. Agreement between the two is reported as Cohen's kappa.
4. **Human review, round 1** (first 577 texts). A team member decided every row where the two annotators disagreed
   (45), plus a random check of rows where they agreed (10 per agreed label, 40; `needs_review`, chosen by
   `labeling/merge_annotations.py`). The random check estimates how often an agreed AI label is wrong: 1 of 40
   was changed.
5. **Batch 4 and round 2.** The first model comparison showed too few demand examples, so 217 more texts were drawn
   (demand-enriched; `make_label_sample.py ... --tag 2`) and labelled by both annotators the same way
   (95.9% agreement). The reviewer decided the 9 disagreements and a random check of 3 agreements per label
   (1 of 12 changed).
6. **Second look at demand vs. other.** The reviewer asked to revisit the demand/other boundary after the rule for
   product-centred recipes was clarified. Every reviewed text where demand was one of the choices (45) was shown
   again with a short decision guide; 22 labels changed (21 other → demand, 1 price → demand). The first-pass
   decisions are kept in `labeling/reviews_first_pass.json` and in the `first_pass_label` column.
7. `labeling/collect_reviews.py` writes `labeling/labels.csv`: `label` is the human decision where a person
   reviewed the row and the agreed AI label elsewhere; `label_source` records which.
