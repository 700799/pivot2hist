# pivot2hist

Auto-fitted pivot tables that toggle to histograms and back, with slicing.
Built for cyber logs (firewall, auth, DNS, EDR ...) but works on any tabular data.

```
pip install pivot2hist            # pandas + numpy only
pip install "pivot2hist[plot]"    # + matplotlib for .plot()
```

```python
import pivot2hist as p2h

v = p2h.fit("firewall.csv")        # a DataFrame, path, list of dicts ... anything tabular
print(v)                           # the best pivot for a 40 x 12 box, chosen automatically
print(v.toggle())                  # the same table as a histogram
print(v.slice(action="deny"))      # sliced (filters stack, and survive toggling)
print(v.histogram("bytes"))        # distribution of one column, nice log bins
```

Or from the shell:

```
pivot2hist firewall.csv
pivot2hist firewall.csv --hist --on bytes --by action
pivot2hist firewall.csv --slice action=deny --slice dst_port=22,3389 --rows src_ip --cols dst_port --count
pivot2hist --demo firewall
```

## What "auto-fit" does

`fit()` decides the **ratio and dimensions** of the pivot for you:

1. **Profiles** every column: `categorical` (labels, including numeric ones like ports or
   severities), `numeric` (quantities), `datetime`, `boolean`, `id` (high-cardinality
   identifiers) or `constant`.
2. **Picks a measure**: the sum of an additive-looking numeric column (`bytes`, `count`,
   `amount`, `revenue` ...), else the row count. Override with `values=` / `agg=`.
3. **Plans level budgets** so the table fits the box (`max_rows` x `max_cols`, default 40 x 12):
   * high-cardinality labels become *top-N + "(other)"*,
   * numeric columns are binned with nice 1/2/2.5/5 widths (log-spaced 1-2-5 bins when the
     data is heavy-tailed, as bytes and durations usually are),
   * timestamps are bucketed at the finest of minute/5 min/15 min/hour/6 h/day/week/month/quarter/year that fits.
4. **Searches** row/column assignments of up to `layers` dimensions per axis (default 2, so
   `src_ip > dst_port` on the rows is possible) and scores each on a sample of the data:
   entropy of the cell counts (many, evenly used cells), mutual information between the axes
   (tables that show structure), minus sparsity, distance from the target `aspect` ratio,
   extra layers, mass hidden in "(other)", and a few pivot-shaped priors (entities such as IPs,
   hosts and users go down the side; small categories go across the top).

```python
p2h.fit(df, max_rows=20, max_cols=6)    # a smaller box
p2h.fit(df, layers=1)                   # single dimension per axis
p2h.fit(df, aspect=4)                   # target rows/cols ratio
p2h.fit(df, rows=["src_ip"], cols=["action"])          # fix the axes, auto-pick the measure
p2h.fit(df, rows=["timestamp"], values="bytes", agg="mean")
p2h.fit(df, rows=[{"column": "bytes", "bins": 6}, "action"], agg="count")   # explicit binning
p2h.fit(df, rows=[{"column": "timestamp", "freq": "h"}], cols=[{"column": "src_ip", "top": 5}])
p2h.fit(df, exclude=["session_id"], bins="fd", scale="linear", order="natural")
```

Example (`p2h.sample.firewall_logs()`, `max_rows=8, max_cols=5`):

```
pivot · sum(bytes) by dst_port (top 7) x severity · 5,000 rows
severity                  1          2        3        4        5
dst_port (top 7)
22                  363,529    410,430   76,948   24,709   11,004
53                  942,479    792,997  638,339  148,153   26,825
80                2,613,591  1,031,321  811,617  312,231  168,930
443               2,837,046  2,901,985  758,086  430,486  570,214
445                 465,295    250,694   53,423   28,583   12,484
3389                 85,024    205,255  183,516   14,443   16,613
8080                431,607    305,602  159,732   30,346   31,373
(other)           1,774,343    846,769  526,193  215,861  148,340
```

## Multi-layer

Both axes can stack dimensions. Levels are planned jointly so the product still fits the box:

```
pivot · count by severity > action x protocol · 5,000 rows
protocol           TCP  UDP ICMP
severity action
1        allow   1,447  339   45
2        allow     890  211   22
         deny      304   18   13
         drop      115    4    5
3        allow     468  120   20
...
```

`v.layers(1)` / `v.layers(3)` refit with a different depth; `v.fit_to(20, 6)` refits into another box.

## Toggle to histogram and back

`toggle()` flips the *same* layout between the two modes: the row axis becomes the bins,
the column axis becomes the series. Toggling again gives the pivot back, with any slices
added in the meantime.

```
hist · sum(bytes) by dst_port (top 7) x severity · 5,000 rows
dst_port (top 7)  1             2             3             4             5
22                ▊       364K  ▉       410K  ▏      76.9K  ▏      24.7K  ▏        11K
53                ██      942K  █▋      793K  █▍      638K  ▎       148K  ▏      26.8K
80                █████▍ 2.61M  ██▏    1.03M  █▋      812K  ▋       312K  ▍       169K
443               █████▉ 2.84M  ██████  2.9M  █▋      758K  ▉       430K  █▏      570K
...
```

`histogram(on, by=None, bins=None, values=None, agg=None, scale=None)` targets a column
instead. Numeric columns get nice bins, timestamps get time buckets, labels get top-N bars.
`toggle()` from such a histogram returns to the pivot it was made from.

```python
v.histogram("bytes")                          # count per nice (log) bin
v.histogram("bytes", bins=8, scale="linear")  # 8 equal-width bins
v.histogram("bytes", by="action")             # one series per action
v.histogram("timestamp")                      # events over time
v.histogram("dst_port", by=["protocol", "action"], values="bytes", agg="sum")
```

```
hist · count by bytes (bins) · 5,000 rows
bytes (bins)  count
[0, 1)        ▋                                                             20
[1, 10)       █████▎                                                       174
[10, 100)     ███████████████████████▎                                     777
[100, 1K)     ██████████████████████████████████████████████████████████ 1.94K
[1K, 10K)     █████████████████████████████████████████████████▉         1.66K
[10K, 100K)   ████████████▎                                                407
[100K, 1M]    ▊                                                             23
```

Bin rules: `auto` (Freedman-Diaconis clamped to [Sturges, 4 x Sturges], safe on heavy tails),
`fd`, `sturges`, `scott`, `sqrt`, `rice`, or an integer.

## Slices

Slices are filters that stack, show up in the title, and carry across toggles.

```python
v.slice(action="deny")                        # equality
v.slice(dst_port=[22, 3389])                  # membership
v.slice(bytes=(1000, 50000))                  # range [lo, hi)
v.slice(bytes=slice(1000, None))              # open range
v.slice(bytes=">= 1000")                      # comparison string: == != > >= < <=
v.slice(src_ip="~^10\\.0\\.1\\.")               # regex (leading ~)
v.slice(timestamp="2026-03-02")               # a whole day / "2026-03" a month / "2026" a year
v.slice(timestamp=("2026-03-02", "2026-03-04"))
v.slice(bytes=lambda s: s > s.median())       # callable mask
v.slice("bytes > 1000 and action == 'deny'")  # pandas query expression
v.slice("action", "deny")                     # positional

v.slices                                      # ['action=deny', 'dst_port∈{22,3389}']
v.unslice("action")                           # drop one; v.unslice() drops all
v.slicers()                                   # {column: [(value, count), ...]} to pick from
```

If a slice collapses one of the layout's dimensions to a single value (slicing
`action="deny"` on a table whose columns are `action`), the layout is refit automatically.
Pass `refit=False` to keep it, or `refit=True` to always refit on the sliced data.

## Views are immutable and cheap

Every call returns a new `View` sharing the same source frame. Tables are computed
lazily and cached per view. The auto-fit scores candidates on a sample (`sample=50_000`
rows by default); the final table always uses every row.

```python
v.pivot()        # pandas DataFrame (observed level combinations only)
v.bins()         # pandas DataFrame with every planned bin (histogram mode)
v.table()        # whichever matches v.mode
v.layout         # Layout(rows=(Dim,...), cols=(Dim,...), values, agg)
v.layout.describe()
v.render(width=120, ascii_only=True, max_rows=20)
v.to_dict(); v.to_json(); v.to_csv("out.csv")
v.plot()         # matplotlib: heatmap for pivots, bars for histograms
v._repr_html_()  # notebooks render both modes
```

## Loading data

`p2h.load()` (used by `fit`/`histogram`) accepts a DataFrame or Series, a file path
(csv, tsv, json, jsonl, parquet, xlsx, feather, optionally gzip/bz2/zip/xz compressed),
a list of dicts, a dict of lists or a 2-D numpy array. Text columns that look like timestamps
and integer columns holding epoch seconds/milliseconds are parsed as datetimes
(`parse_dates=False` to skip).

## CLI

```
pivot2hist FILE [--hist] [--on COL] [--by COL,COL] [--bins N|rule] [--scale auto|linear|log]
                [--rows COL,COL] [--cols COL,COL] [--values COL] [--agg sum|mean|...] [--count]
                [--max-rows N] [--max-cols N] [--layers N] [--aspect R]
                [--slice COL=VAL]... [--where EXPR]
                [--json | --csv | --layout | --profile | --slicers] [--width N] [--ascii]
pivot2hist --demo firewall|auth
cat data.csv | pivot2hist -
```

`--slice` forms: `col=val`, `col=a,b,c`, `col=lo..hi`, `col>=val`, `col!=val`, `col~regex`.

## Options

| option | default | meaning |
| --- | --- | --- |
| `max_rows`, `max_cols` | 40, 12 | the box the pivot must fit |
| `layers` | 2 | max stacked dimensions per axis |
| `aspect` | box ratio | target rows/cols ratio (explicit values are weighted strongly) |
| `max_sparsity` | 0.6 | empty-cell fraction tolerated before a layout is penalised |
| `max_bins` | 30 | bin cap for `histogram()` |
| `bins` | `"auto"` | bin rule for numeric dimensions |
| `scale` | `"auto"` | `"linear"` / `"log"` numeric bins |
| `order` | `"auto"` | level order: natural for numbers/time, by frequency for text |
| `sample` | 50000 | rows used to score candidates |
| `search_width` | 7 | dimension candidates entering the search |
| `exclude` | `()` | columns never used |
| `max_categories`, `discrete_max`, `id_ratio` | 50, 20, 0.5 | profiling thresholds |

## Development

```
pip install -e ".[dev]"
pytest
```
