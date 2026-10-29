# Data dictionary and provenance

## 1. Sources

| | Reddit (via Arctic Shift) | USDA AMS terminal market prices | USDA AMS movement (shipments) | Google Trends (search interest) |
|---|---|---|---|---|
| What | Posts and comments that mention avocados, lettuce, onions or cucumbers | Daily wholesale price reports for fresh produce at 10 US terminal markets | Daily pounds shipped from US production districts and imported through ports of entry, by commodity | Weekly US search interest for four terms per product: the product name alone and with "price", "shortage" and "recall" (`config/products.yaml`) |
| Access | Arctic Shift archive API, `https://arctic-shift.photon-reddit.com` (no key) | MyMarketNews MARS API, `https://marsapi.ams.usda.gov/services/v1.2/reports` (free API key) | Same API; the "Truck, Air and Boat Daily Movement" report of every Market News office (`config/ams_markets.yaml`) | Google Trends, `https://trends.google.com`, through the pytrends library (no key; Google offers no public API) |
| Access date | 26–27 October 2026 | 26 October 2026 | 26 October 2026 | 27 October 2026 |
| License / terms | Public Reddit content, archived by the Arctic Shift project for research; use is subject to the Reddit User Agreement and Public Content Policy. We keep only the text needed for the analysis and replace user names by pseudonymous ids. | US Government work, public domain; data from USDA AMS Market News. | Same as prices | Google Trends data, used under Google's terms of service with attribution (source: Google Trends) |
| Coverage | 1 Jan 2022 – 31 Dec 2024; 29 topical subreddits (consumers, retailers, food service, growers, economy) and 9 city subreddits listed in `config/sources.yaml`, plus title search in r/news | 1 Jan 2022 – 31 Dec 2024; Atlanta, Baltimore, Boston, Chicago, Columbia SC, Detroit, Los Angeles, Miami, New York, Philadelphia | 1 Jan 2022 – 31 Dec 2024; all US shipping districts and ports of entry covered by Market News; some offices stopped reporting during the period | 1 Jan 2022 – 31 Dec 2024, 157 weeks, 16 search terms |
| Update frequency | Archive updated continuously (scores settle ~36 h after posting); monthly dumps | Daily (business days) | Daily (business days), late additions flagged | Continuous; a 3-year request returns one value per week |
| Geography | Mostly unknown; attached when the subreddit is a city or regional grocery chain, or one US state is named in the text | Market city | Shipping district or port of entry, origin country | United States as a whole |

## 2. Transformations

1. **Collection** (`data_collection/collect_reddit.py`): every post of each subreddit in the study period is listed
   page by page; posts whose title or body matches a product regex (`config/products.yaml`) are kept; weekly
   totals of all posts in the topical subreddits are kept as the denominator for normalised volume. The 9 city
   subreddits are listed the same way (step `geo`) to give posts a location; they are left out of the denominator.
   For posts naming a product in the title and with ≥3 comments, the comment tree is downloaded and comments
   mentioning a product are kept. r/news is searched by title keywords instead of listing.
2. **Cleaning** (`data_collection/build_event_dataset.py`), counts in `data/processed/filter_log.csv`: bots and
   automated messages, moderator notices, deleted/removed text, NSFW posts, crossposts, self-promotion and paid
   promotion, link farms (3+ URLs), texts under 15 characters, authors repeating the same text 3+ times, and
   duplicate texts are removed.
3. **Event tagging**: keyword dictionary (`config/event_keywords.yaml` → `data/processed/keyword_dictionary.csv`).
4. **Prices** (`data_collection/collect_ams.py`, `analysis/prices.py`): price per line = midpoint of the "mostly"
   range if reported, else of the low–high range; weekly and daily fixed-composition indices from a two-way
   fixed-effects model (item series × time).
5. **Shipments** (`data_collection/collect_ams_movement.py`): movement lines of all offices are combined, late
   additions are moved to the shipping date they belong to, lines repeated in two offices' reports are kept once,
   and pounds are summed per product and ISO week, split into domestic shipments and imports.
6. **Search interest** (`data_collection/collect_google_trends.py`): each term is requested on its own, US only,
   over the study period. Google scales each request to 0–100 (100 = the term's busiest week; 0 = too few searches
   to report). Google weeks run Sunday to Saturday; each is assigned to the ISO week starting the next Monday.

## 3. `data/processed/event_dataset.csv`

One row per (text, product) pair.

| Column | Meaning |
|---|---|
| `obs_id` | Observation id: `p_<post id>_<product>` or `c_<comment id>_<product>` |
| `record_id` | Reddit record id (`p_`/`c_` + Reddit id) |
| `source` | Always `reddit` |
| `record_type` | `post` or `comment` |
| `collection_method` | `subreddit_listing`, `city_listing` (city subreddits), `title_search` (r/news), `comment_tree` |
| `subreddit`, `subreddit_group` | Subreddit and its group (consumer, retail, foodservice, growers, news, city) |
| `thread_id` | Reddit id of the post the record belongs to (the post itself for posts); used for grouped evaluation |
| `created_utc`, `date`, `week_start` | Unix time, UTC date, Monday of the ISO week |
| `product` | `avocado`, `lettuce`, `onion`, `cucumber` |
| `n_products_mentioned` | Number of study products in the same text |
| `text` | The post or comment itself (post: title + body; comment: body), as written |
| `event_types` | All market-event types whose keywords occur (`|`-separated) or `none` |
| `primary_event_type` | First match in the order food_safety_recall > weather_event > supply_shortage > price_change > marketing_promotion > demand_change; `other` when none |
| `geo_city`, `geo_state`, `geo_source` | Geography when available and how it was obtained |
| `ams_market` | AMS terminal market in the same city (city subreddits only) |
| `score`, `num_comments` | Reddit score; number of comments (posts) |
| `author_id` | Pseudonymous author id (`U000123`), `deleted` for deleted accounts |
| `url` | Link to the post or comment |

## 4. `data/processed/ams_terminal_prices.csv.gz`

One row per reported price line: `report_date`, `market`, `market_location_city/state`, `slug_name` (report id),
`product`, `commodity`, `variety`, `package`, `item_size`, `origin`, `organic`, `low_price`, `high_price`,
`mostly_low_price`, `mostly_high_price` (US$ per package), `market_tone_comments`, `offerings_comments`.

## 5. `data/processed/ams_shipments_weekly.csv`

One row per product and ISO week (`week_start`, Monday): `domestic_lb`, `import_lb`, `total_lb` = pounds shipped
nationally. Line-level data: `data/processed/ams_movement_daily.csv.gz` (`ship_date`, `office`, `trans_Mode`,
`commodity`, `district`, `origin`, `import`, `pounds`, ...).

## 6. Other files

| File | Content |
|---|---|
| `data/processed/keyword_dictionary.csv/.json` | Product and market-event keyword dictionary |
| `data/processed/summary_table.csv/.md` | Observations per product and event type |
| `data/processed/filter_log.csv` | Records removed by each cleaning rule |
| `labeling/label_sample.csv` | Texts drawn for labelling, with the batch each came from (`labeling/make_label_sample.py`); `upgrade = True` marks the 340 silver texts chosen to become gold (`labeling/select_upgrade.py`) |
| `labeling/labels.csv` | Final labels (price / supply / demand / other). `tier = gold`: 1,458 texts, used for training and for every accuracy figure (1,118 labelled in the first rounds with every AI disagreement decided by a person, plus 340 silver texts moved up after a person confirmed or corrected them, or ChatGPT agreed with both Claude annotators); `tier = silver`: 1,488 texts where two AI annotators agreed, not used by the final model (55 rows where they disagreed have no label). Columns record both AI labels and reasons, review round, first-pass and re-checked human label, adjudication, ChatGPT's round-2 label (`label_chatgpt_round2`) and `label_source` |
| `labeling/annotations/annotator{1,2}_*.jsonl` | Labels and one-line reasons from the two Claude annotators (Opus, Sonnet), per batch |
| `labeling/annotations/annotator3_codex.jsonl`, `annotator4_chatgpt.jsonl` | Independent blind labels of the 1,118 gold texts from OpenAI Codex and ChatGPT (`labeling/codex_task/TASK.md`) |
| `labeling/adjudication.json`, `adjudication_sample.csv` | Blind A/B decisions on the 73 gold texts where Codex and ChatGPT agreed with each other but not with the first label (round 1), and on the 27 moved-up texts where ChatGPT disagreed with our label (`round = 2`) |
| `labeling/crosscheck2/`, `adjudication2_sample.csv`, `annotations/annotator4_chatgpt_round2.jsonl` | ChatGPT's blind labels of the 340 moved-up texts (task, inputs and outputs) and the round-2 conflicts shown for blind decision |
| `labeling/collab_sample.csv`, `collab_translations.jsonl` | 80 texts for the second human annotator and their Chinese translations (reading aid only) |
| `outputs/tables/annotator_agreement.csv` | Pairwise agreement and Cohen's κ between annotators, Fleiss' κ for all four |
| `outputs/tables/human_agreement.csv` | Agreement between the two human annotators on the 80-text sample |
| `outputs/tables/model_cv_metrics.csv`, `model_cv_predictions.csv`, `selected_model.json` | Cross-validated accuracy of every classifier and the one selected |
| `outputs/tables/label_prevalence.csv` | Label shares per product: labelled sample, model, and random samples with 95% intervals |
| `data/processed/classified_observations.csv.gz` | Model label, class probabilities and sentiment for every observation |
| `outputs/tables/*.csv` | Weekly signals, anomalies, price indices, lead/lag and event tables |
| `config/events_reference.csv` | Documented market events with dates and news sources |
| `data/processed/google_trends_weekly.csv` | One row per product, search term and week: `product`, `series` (`total`, `price`, `shortage`, `recall`), `term`, `week_start` (Monday), `google_week_start` (Sunday), `interest` (0–100 within the term), `is_partial` |
| `data/raw/google_trends/<term>.csv` | Each term as downloaded |

## 7. Limitations

- Reddit users are younger, more urban and more online than US shoppers overall; subreddit choice shapes what is
  seen. Volumes are therefore treated as relative signals, not as estimates of national demand.
- Keyword matching misses misspellings and slang and can include incidental mentions (onions in recipes); the
  classifier's **other** class absorbs most incidental mentions.
- Arctic Shift may miss content removed before archiving.
- Most observations have no location; regional comparisons use the subset that has one.
- AMS terminal prices are wholesale prices for specific packs and grades; retail prices move later and less.
- Movement reports cover shipments that Market News offices track; small districts, local sales and offices
  that stopped reporting are missing, so shipments are a relative measure of quantity, not a census.
- Google Trends values are relative within each term and based on a sample of searches, so a new download can
  differ slightly from the committed one. The shortage and recall terms have too few searches in most weeks and
  are 0 there; their spikes rest on weeks with unusual search activity.

## 8. Forecast check, 2019–2026 (`forecast_check/`)

The forecast check asks whether online signals help forecast national wholesale prices. It extends the main
pipeline to 1 January 2019 – 30 September 2026 and adds three sources.

| | GDELT DOC 2.0 (news) | openFDA food enforcement (recalls) | Wikipedia pageviews |
|---|---|---|---|
| What | Daily number of US English-language online news articles that name the product, alone and with shortage, recall or price words; their share of all US articles; average tone | Number of food recalls whose product description or reason names the product, by recall start date | Daily views of the English Wikipedia article on each product (users only) |
| Access | `https://api.gdeltproject.org/api/v2/doc/doc` (no key) | `https://api.fda.gov/food/enforcement.json` (no key) | `https://wikimedia.org/api/rest_v1/metrics/pageviews` (no key) |
| Access date | 28 October 2026 | 28 October 2026 | 28 October 2026 |
| License / terms | Free and open for any use with citation of the GDELT Project | Public domain (CC0), US FDA | CC0, Wikimedia Foundation |
| Coverage | 1 Jan 2019 – 30 Sep 2026, daily, 16 queries (`forecast_check/collect_extra.py`) | Same period | Same period |

Reddit, AMS prices, AMS shipments and Google Trends for 2019–2021 and January 2025 – September 2026 were collected
on 28 October 2026 with the main pipeline, unchanged except for the study period (`forecast_check/build_period.sh`).
New texts were classified with the model trained on the 2022–2024 labels; no new labels were made. Google Trends
values are scaled within each download, so the forecast check standardises them within each period.

| File | Content |
|---|---|
| `forecast_check/data/extra_daily.csv` | One row per source, product, topic, measure and day: `source` (`gdelt`, `fda`, `wikipedia`), `product`, `topic`, `resolution`, `measure` (`TimelineVolRaw` = article count with `norm` = all US articles that day; `TimelineTone`; `recalls`; `views`), `date`, `value`, `norm` |
| `forecast_check/data/panel_long.csv` | One row per product and week, 2019-01 to 2026-09: national log price index, shipments, the Reddit weekly signals of the main analysis, Google Trends (`g_segment` = download period), GDELT (`news_v_*`, `news_share_*`, `news_tone_*`), `fda_recalls`, `wiki_views` |
| `outputs/tables/forecast_long_*.csv` | Forecast check results (`forecast_check/predict_long.py`) |
