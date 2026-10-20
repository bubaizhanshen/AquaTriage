# Publish AquaTriage on Streamlit Community Cloud

The hosted entrypoint is `app/cloud.py`. It loads the bundled application
reference and supports case uploads, prediction, ranking, and result download.
It does not fit models on the public server. The local entrypoint,
`app/review.py`, also supports fitting a reference from the user's measurements.

## Create the public app

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and
   authorize access to the GitHub repository.
2. Choose **Create app**, then supply:

   | Setting | Value |
   | --- | --- |
   | Repository | `bubaizhanshen/AquaTriage` |
   | Branch | `main` |
   | Main file path | `app/cloud.py` |
   | Python version, under Advanced settings | **3.11** |
   | Secrets | None required |

3. Choose an available app subdomain and deploy. The subdomain is not reserved
   by this repository. Do not announce an address until deployment succeeds.
4. In app settings, check that sharing is public.
5. Open the generated URL in a signed-out browser. Run the included example,
   download the ZIP, and test a small uploaded CSV. Check both objectives and
   a mobile viewport before sharing the address.

After successful verification, add the actual public URL to the repository
description and README. A local `127.0.0.1` address is not a public app URL.

## Deployment contents

- `app/requirements.txt`: pinned Python dependencies, colocated with the
  entrypoint so Community Cloud selects them.
- `packages.txt`: the LightGBM OpenMP runtime.
- `.streamlit/config.toml`: UI and server settings; CORS and XSRF protection stay on.
- `models/review-v1/`: fitted reference and checksum/configuration manifest.
- `data/examples/review_queue.csv`: 23 real input cases for 12 chemicals,
  without measured toxicity outcomes.
- `src/ecoood/`: the same feature, prediction, and ranking code used locally.

The deployment needs no repository token or API key in the application.
Authentication to publish/manage the app is performed by the owner in the
hosting service, not by visitors.

## Resource and data handling

The hosted upload limit is 2 MB and 1000 cases per run. Computation is serialized
within an application process to limit concurrent memory use. These are hosting
limits, not scientific inclusion criteria. For a larger chemical queue, use the
local CLI with the whole queue; splitting a queue changes the review quota and
selection problem.

Uploaded tables and results remain in the user's server session, not in a
shared result cache or saved data file. The app does not send chemical inputs
to third-party prediction APIs. A public hosting provider still processes the
data on its infrastructure; confidential inputs should be run locally.

The public model is cached across sessions. Molecule images have a bounded
one-hour cache. No user-supplied executable model file is loaded.

The cloud entrypoint directs application temporary files to `tmp/cloud/`.

## Verify before changing the model

Keep model/library versions aligned. Refit with the new environment if a
library upgrade changes the model serialization or preprocessing. Never
disable the compatibility check simply to make an old artifact load.

Compare the downloaded `chemicals.csv`, `endpoints.csv`, and `cases.csv` with
the local CLI for the same reference, input, task, and quota. Their results
should agree. Hosting is an access mechanism, not an additional validation
dataset.

See the official [deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [dependency instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
