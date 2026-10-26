# Data dictionary and provenance

## 1. Sources

| | Reddit (via Arctic Shift) | USDA AMS terminal market prices | USDA AMS movement (shipments) |
|---|---|---|---|
| What | Posts and comments that mention avocados, lettuce, onions or cucumbers | Daily wholesale price reports for fresh produce at 10 US terminal markets | Daily pounds shipped from US production districts and imported through ports of entry, by commodity |
| Access | Arctic Shift archive API, `https://arctic-shift.photon-reddit.com` (no key) | MyMarketNews MARS API, `https://marsapi.ams.usda.gov/services/v1.2/reports` (free API key) | Same API; the "Truck, Air and Boat Daily Movement" report of every Market News office (`config/ams_markets.yaml`) |
| Access date | 26–27 October 2026 | 26 October 2026 | 26 October 2026 |
| License / terms | Public Reddit content, archived by the Arctic Shift project for research; use is subject to the Reddit User Agreement and Public Content Policy. We keep only the text needed for the analysis and replace user names by pseudonymous ids. | US Government work, public domain; data from USDA AMS Market News. | Same as prices |
| Coverage | 1 Jan 2022 – 31 Dec 2024; 29 topical subreddits (consumers, retailers, food service, growers, economy) and 9 city subreddits listed in `config/sources.yaml`, plus title search in r/news | 1 Jan 2022 – 31 Dec 2024; Atlanta, Baltimore, Boston, Chicago, Columbia SC, Detroit, Los Angeles, Miami, New York, Philadelphia | 1 Jan 2022 – 31 Dec 2024; all US shipping districts and ports of entry covered by Market News; some offices stopped reporting during the period |
| Update frequency | Archive updated continuously (scores settle ~36 h after posting); monthly dumps | Daily (business days) | Daily (business days), late additions flagged |
| Geography | Mostly unknown; attached when the subreddit is a city or regional grocery chain, or one US state is named in the text | Market city | Shipping district or port of entry, origin country |

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
| `description` | First 280 characters of the text (whitespace normalised) |
| `text` | Full text (post: title + body; comment: body) |
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
| `labeling/labels.csv` | Hand-labelled sample (price / supply / demand / other) |
| `data/processed/classified_observations.csv.gz` | Model label, class probabilities and sentiment for every observation |
| `outputs/tables/*.csv` | Weekly signals, anomalies, price indices, lead/lag and event tables |
| `config/events_reference.csv` | Documented market events with dates and news sources |

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
