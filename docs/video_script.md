# Video Presentation Script — Team P
## "The Green Premium: Correlating Urban Real Estate Prices, Air Quality, and Public Green Spaces in Dublin"
### National College of Ireland — MSc in Data Analytics | Max 10 minutes

---

## Segment 1 — Introduction (1:00)

*[Show title slide with project name, team names, student numbers]*

"Welcome. I'm [Name], and with my team members [Name] and [Name], we investigated a question that matters both to property buyers and urban planners: **do Dublin homes near parks and clean air cost more?**

Dublin is an ideal study. It's a fast-growing European capital with one of the lowest ratios of park area to population in Europe — yet some postcodes have enormous green spaces like the Phoenix Park. It also has one of the most transparent property transaction registers in the EU.

We integrated three datasets: **92,254 residential transactions** from the PSRA Property Price Register covering 2015 to 2020, **624,000 air quality readings** from the Dublin City Council Sonitus monitoring network, and **1,000 green space features** from DCC open data.

Our five research questions are: [read from slide] Is there a green premium? Does air quality correlate with price? Do they interact? How has this changed over time? And can we predict price from these variables?"

---

## Segment 2 — Architecture Walkthrough (2:00)

*[Show architecture diagram — docs/architecture.png or the ASCII diagram from CLAUDE.md]*

"Let me walk you through the technical pipeline.

**Data ingestion** feeds three raw stores. Property CSVs go into **PostgreSQL 16** with PostGIS — we chose Postgres because tabular transactions need SQL joins and the PostGIS extension enables native spatial indexing. Air quality JSON from the Sonitus REST API and green space GeoJSON go into **MongoDB 7** — document stores are ideal for semi-structured, schema-varying data.

The Sonitus API is a POST-based API requiring credentials as query parameters, not Basic Auth as its documentation implies. It also enforces a 7-day maximum window per request. Our ingest chunks each calendar month into five API calls, caches responses as JSON files, and upserts into MongoDB with a compound unique key on station, timestamp, and pollutant — making the ingest fully **idempotent**.

**ETL** runs in four steps. Property addresses are geocoded through a three-tier strategy: SQLite cache, then Nominatim OSM at 1 request per second, then postcode centroid fallback. All spatial joins happen in EPSG:2157 — Irish Transverse Mercator — for accurate metric distances. We use `geopandas.sjoin_nearest` for park distance and a `cKDTree` for air quality station matching.

The enriched output lands in both PostgreSQL's `processed.property_enriched` table and a MongoDB mirror, satisfying the requirement to use both databases pre- and post-processing."

---

## Segment 3 — Code Highlights (2:00)

*[Switch to VS Code or screen share showing the three key code sections]*

**Geocoding (src/etl/geocode.py)**

"The geocoding cache is a SQLite database at `data/interim/geocode_cache.db`. Every result — success or failure — is written back to cache so the same address is never looked up twice. The three tiers are: cache lookup is instant, Nominatim is 1 second per call with retry logic, and the postcode centroid dictionary has 22 Dublin districts hand-coded. This reduces a potential 25-hour Nominatim run to under 2 minutes."

**Spatial Join (src/etl/join.py)**

"The green area buffer uses `geopandas.sjoin_nearest` for park distance — one line of code with a spatial R-tree index. For buffer area, we compute polygon intersections. For air quality, we build a KD-tree over station ITM coordinates and query each property's nearest station, then look up the annual mean for that station's year."

**OLS Regression (src/analysis/regression.py)**

"The OLS formula is: `log_price ~ log1p(nearest_park_dist) + green_area_500m + mean_no2 + C(year_of_sale) + C(construction)`. We use HC3 heteroscedasticity-robust standard errors from statsmodels. The year fixed effects control for the Dublin property boom, which pushed median prices from €300K in 2015 to €365K in 2019."

---

## Segment 4 — Dashboard Demo (3:00)

*[Open browser at localhost:8050]*

"The dashboard has six tabs. Let me demo each briefly.

**Overview tab** — Four KPI cards show headline metrics: 92,254 total sales, median price €320,000. On the left, the log-price histogram confirms the log-normal distribution that justifies our log-transform. On the right, the price vs park distance scatter — note the trendline slopes up, which I'll explain in findings.

*[Switch year slider to 2019–2020]*

The year slider is a global filter — all charts update. You can see 2020 has fewer transactions due to COVID restrictions reducing market activity.

**Geographic tab** — [drag to Dublin map] Properties coloured by price on an OpenStreetMap base layer. Zoom into Dublin 4 — high prices cluster around Herbert Park. Dublin 15 and Tallaght show lower prices with different park configurations.

*[Switch pollutant dropdown to NO₂]*

Switching to NO₂ colours by air quality. The inner-city corridor near the quays shows the highest readings — Civic Centre and Winetavern Street stations.

**Green Premium tab** — [show F3 boxplot] Price clearly increases across green-area quintiles. The top 20% by green area has a median price ~15% above the bottom 20%. [show F6 temporal chart] The gap widens from 2015 to 2019, then narrows slightly in 2020.

**Air Quality tab** — NO₂ scatter with trendline. Note the positive slope — I'll explain the confounding shortly.

**Statistical Model tab** — [show F8 forest plot] The OLS forest plot shows green area and park distance with statistically significant coefficients. The year 2020 fixed effect is the largest: COVID suppressed transactions, raising the selection-bias risk.

**Download CSV button** — [click] exports the currently filtered dataset. Works with any combination of year and postcode filters."

---

## Segment 5 — Findings Discussion (1:30)

"Three non-arbitrary findings:

**Finding 1 — Green area premium is real and robust.** Green area within 500 metres has a Pearson r of 0.095 with log-price, significant at p<0.001 across 92,000 observations. The OLS estimate implies a one-standard-deviation increase in green area corresponds to approximately a 16% price premium. This replicates Conway et al.'s US findings in an Irish context.

**Finding 2 — Park distance is confounded by spatial heterogeneity.** The raw correlation between distance and log-price is negative — closer to parks means higher price. But the OLS slope is positive. This isn't a contradiction: it reflects that Dublin 4 and 6, with the highest prices, have large parks at moderate distances from many homes, while inner-city Dublin 1 and 7 have small parks close to lower-priced properties. OLS without spatial fixed effects at the electoral-division level cannot untangle this.

**Finding 3 — Air quality needs causal design.** Raw Pearson r with NO₂ is negative: higher pollution, lower price. But OLS flips positive due to the urban density confound. This is precisely what Chay and Greenstone warned about: you need exogenous variation — like a monitoring station opening — to identify the causal effect. The honest interpretation is that our data is consistent with an air quality discount but the OLS alone doesn't establish it."

---

## Segment 6 — Conclusion (0:30)

"To conclude: we built a fully reproducible pipeline from open-data ingestion through spatial enrichment to interactive visualisation. Green space proximity shows a statistically significant and economically meaningful association with Dublin property prices. Air quality results are directionally consistent with the literature but require a stronger identification strategy.

Limitations: postcode-centroid geocoding loses address-level precision; air quality data is sparse for 2015–2017; no structural controls. Future work: address-level geocoding, difference-in-differences around sensor installations, 2022–2024 data.

The full pipeline, dashboard, and data are reproducible from a single `docker-compose up` command. Thank you."

---

## Recording Tips

- Total target: 10 minutes. Rehearse once with a timer.
- Screen resolution: 1920×1080 minimum. Dashboard at 100% zoom.
- Use OBS or Windows Game Bar (Win+G) to record screen.
- Start dashboard before recording: `python -m src.dashboard.app`
- Ensure Docker containers are running: `docker-compose up -d`
- Export as MP4, H.264, 1080p. File name: `TeamP.mp4`.
