# CHANGELOG

<!--next-version-placeholder-->

## v0.3.0 (2024-11-06)

### Documentation

* docs: updated examples and readme ([`67da2d9`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/67da2d959076f967782c8db7424cee09c3cb3fce))

* docs: added info into euler docstring ([`5755815`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/57558159b788cde729a99141a50d9800f086d37b))

### Feature

* feat: added interface to use custom parameter functions - code-clean-up ([`69ccefd`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/69ccefde99dea83ce6e53720c34787ae3fb0a5ea))

### Style

* style: using f-strings ([`a68976e`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/a68976e56a9bf92c543a9455f734cce4a7a08c48))

* style: replaced code fragment ([`c89c767`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/c89c767bba0be11da965acfa5d13ca4326d534b0))


## v0.2.9 (2024-09-11)

### Documentation

* docs: fixed error in doc config ([`517c532`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/517c532c89941463ff3136b1697f8e43c1b44ba4))

* docs: fixed error in doc config ([`b9d11ab`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/b9d11ab39e030026cda7e341d0b5c9e74fff4cab))

### Fix

* fix: set minimum event gap as maximum of 4 hours and series interval ([`56558c5`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/56558c50f2ed9c9aa499cd8221270aa99f019907))

* fix: added interval to event duration ([`b16e42e`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/b16e42e16a469c94da74d55cb8b8e0b0c1aeba37))

* fix: added print function for idf-parameters ([`b435ccd`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/b435ccdca1bfa747dbf6c797c78dd771f658e632))

### Style

* style: inline loop to multiline loop ([`9c2a92c`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/9c2a92c8d3d050f89a59ab79045e001bd5e5cd52))


## v0.2.8 (2024-08-12)

### Documentation

* docs: minor formatting ([`0fe8471`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/0fe84710b094c3cace609aaa36cf55f18d790b88))

* docs: updated Changelog link for pypi ([`f29346b`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/f29346bd2844265ec0923513f868ec4bfb3976f2))

### Fix

* fix: limit duration steps based on series frequency and skip parameter calculation for durations ranges lower than series frequency. ([`22e9a19`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/22e9a19b7401ccbae97b98d70fa6051957481c81))


## v0.2.7 (2024-07-22)

### Documentation

* docs: update jupyter notebook ([`70d8713`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/70d8713062adb6995ade19ce84fd449bb63e4800))

* docs: added more context in some docstrings ([`16c7568`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/16c7568feb5ac707c9c9294bad298799cf5ddabd))

* docs: pycharm jupyter update ([`e4408f9`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/e4408f9cba32619414e7338c3bd375bbea5ebd6d))

### Fix

* fix: added fix if not pandas timedelta ([`c962733`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/c9627339712dfbe202193fe6b9f382ef50471122))

### Style

* style: split some functions in two separate ones. ([`fbcdaa6`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/fbcdaa6d0cb9f698b2761e3ed403761075350198))

* style: adapt to new numpy version 2.0 ([`c893cef`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/c893cef7da8dce3a0fc3fd552b5257dac1cd8824))


## v0.2.6 (2024-05-16)

### Fix

* fix: fixed error for rain bar plot with only one entry. ([`2384e20`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/2384e20d35c0367ceea520f61ff86ee2748ea245))

### Style

* style: prevent pandas warning ([`058a711`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/058a711a36998d1fb0bc44ee198615c8f660b1fe))


## v0.2.5 (2024-05-13)

### Documentation

* docs: added python example usage in readme ([`15e13b4`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/15e13b4174387b54bb0d21a6759d78b257de988e))

* docs: added python example usage in readme ([`db937a5`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/db937a55b9dea4468a8f1d05e06b675666272ffe))

* docs: added citation information ([`fc8b86b`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/fc8b86bb9c08d2b7e86054e541d1672bd5358234))

* docs: added source of example time series ([`26f734c`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/26f734c06f7b0e374518b3b081ec692754825b39))

### Fix

* fix: add option to not add units to idf table index names &amp; added function add_max_return_periods_pre_duration_to_events ([`1745fe8`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/1745fe8699e846c0ff962bf4cc75e9e19beec321))

### Style

* style: prevent matplotlib warning ([`2a1f1fc`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/2a1f1fc66a602d002f272f846fae92fc410aad7b))

* style: pandas deprecation warning ([`8d4acba`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/8d4acbab2ed73a1dd7b25d696ef4adda039e1f3d))

* style: fixed pandas deprecation errors for frequency strings ([`fb78d8a`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/fb78d8a346473e478cd08d33c76d560008fca902))

* style: fixed pandas deprecation warning ([`b2b700e`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/b2b700e3495622975423d15c99293e7dcdc00422))

## v0.2.4 (2024-01-24)

### Documentation

* docs: no gitter ([`54568ce`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/54568ce7d59e76d02100b9d5c2ef7a8a549e4e49))

### Fix

* fix: new function &gt; add_max_intensities_to_events ([`a4876fb`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/a4876fbb571a1c45495f1db8ed79ab70910329f3))

### Style

* style: plot example ([`195bf70`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/195bf709aa474a1af78070b85006c1a618e4f071))

* style: literal colors ([`4ca8832`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/4ca8832e316558a86ce9465af29ab8bf697d5912))

## v0.2.3 (2023-11-07)

### Fix

* fix: pandas 2.1.2 deprecation warning ([`d9e26c8`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/d9e26c841582b9af7d30d0b3723e061cdfe3cf98))

### Misc.

* prepare Schmitt's ortsunabhaengiger Index ([`e479076`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/e4790766958ab0bdc2fb375699fa62498b917097))


## v0.2.2 (2023-08-07)

### Documentation

* docs: fix changelog ([`1fbe71f`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/1fbe71f6472e02a1a22f9388f38c1a8eacb5d925))

### Fix

* fix: added separating lines to idf bar axes for Short-term and medium-term durations ([`9a69e72`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/9a69e728f081b49a2ce64cf75a3ffef1af56fedf))


## v0.2.1 (2023-08-07)

### Ci

* ci: fix minor and outdated commands ([`4ba117e`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/4ba117e77ae0e9e9303eea0b0f0dad33b7d330bf), [`a39daa8`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/a39daa859fff751cfbbe3947aa72432922bb293b),
[`7bf5dc2`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/7bf5dc2b6c1662e139558fd3d4e0c55777d98b66),
[`5a4e905`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/5a4e9050bbe8231561e5a8d42e2c3f866400c682))

### Fix

* fix: don&#39;t balance parameters for single formulation ([`9525fa6`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/9525fa6df7852451b69904d6eace1a9fa638272f))

### Style

* style: remove tight layout calls ([`6f0e0b2`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/6f0e0b222d85a2ee855a777e5d6aef088b4ab2f0))


## v0.2.0 (2023-08-05)

### Feature

* feat: added ability to use custom functions for parameter formulation. default implementations are linear, log-normal, double-log-normal and hyperbolic. ([`f16f674`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/f16f674eff74dc4fa4c78836684fb66bb3b28f87))


## v0.1.16 (2023-07-27)

### Fix

* fix: retry github ci ([`3ab440a`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/3ab440a0a7f326f45694661ce217c35d9f575ef6))

* fix: retry github ci ([`5cf5b73`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/5cf5b734b3eda35f56f3ec3bbd301b4f9c08b1b9))

* fix: retry github ci ([`796982a`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/796982adaeab9627f6ac6dbf0179f1f4c08100c6))

* fix: publishing to pypi ([`4927caa`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/4927caad9028252a42f34d89f87f7c50f5e24448))

* fix: publishing to pypi ([`ab98300`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/ab983007c59e1397d598570409a89fd0c612d855))

* fix: gihub actions ([`8d227af`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/8d227afd15fa2aade11c69af0a6c09f730fb6ffc))

* fix: github CI release fix ([`d39dd08`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/d39dd08d30c3eee5f76e8f0d91600989ec6d2fd0))

* fix: added duration steps parameter for outputs ([`6d2c2d4`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/6d2c2d4820761d0266357b7d5ed07a66ffbee7cb))

* fix: added german event plot caption ([`282a8e4`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/282a8e43737685c04b7798afb772d589579d9146))

* fix: added ability to use an index with a timezone ([`c5f396e`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/c5f396eb35d20ada67442879dbb59f3daf368487))

### Misc.

* added points to example plot ([`8578219`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/85782194d149c9b8f75ac3df55b08b9178788ebf))

* enable analysis for index with timezone ([`ed5b90e`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/ed5b90e5956ce59871c8a107f33fd3b0f91beef3))

* minor changes in example ([`c11ff4a`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/c11ff4a99eeb83caeb024e4a4ffd8b2a72a47dcf))

* start attempt to rewrite parameter formulation ([`55a3990`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/55a39904f4ad8c725610b1958b9746f0ff9cff19))


## v0.1.15 (2023-04-13)

### Fix

* fix: make tqdm and matplotlib optional ([`030994f`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/030994fec09b39d703da2363b289a2efd0281f7d))


## v0.1.14 (2023-03-07)

### Fix

* fix: added warning for synthetic rain series if height &lt; 0 ([`1cca9c2`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/1cca9c20fd21225a06d9c96efe3b73139c30bee3))

### Style

* style: minor ([`114c88d`](https://github.com/MarkusPic/intensity_duration_frequency_analysis/commit/114c88d16eb6806251e87b958c5c63f3c317d801))
