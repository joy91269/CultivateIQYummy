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
6. When in doubt between a signal and **other**, choose **other**. Use the `notes` column for hard cases.

## Procedure used in this project

1. `labeling/make_label_sample.py` drew a stratified sample (product × keyword-candidate), one text per thread.
2. A first-pass label was proposed for every row by an AI assistant (Claude) following this codebook.
3. A team member reviewed every row in the review page, accepting or correcting the proposed label.
   The column `label` holds the final human-reviewed label; `proposed_label` holds the first pass, so the
   agreement between the two is reported as well.
