# francois141.github.io

Personal website of François Costa, built with the [al-folio](https://github.com/alshedivat/al-folio) Jekyll template (v1.x).

## Where things live

| Content                    | File                                   |
| -------------------------- | -------------------------------------- |
| Site settings              | `_config.yml`                          |
| Home page (bio)            | `_pages/about.md`                      |
| Publications               | `_bibliography/papers.bib`             |
| Social links               | `_data/socials.yml`                    |
| Blog posts                 | `_posts/` (`published: false` hides a post) |
| Profile picture            | `assets/img/prof_pic.jpg`              |

Layouts, styles and scripts come from the `al_folio_*` gems pinned in the `Gemfile`, not from this repository.

## Running locally

```bash
bundle install
bundle exec jekyll serve --livereload
```

Then open <http://localhost:4000>. Restart the server after editing `_config.yml`.

With Docker instead: `docker compose up`, then open <http://localhost:8080>.

## Deployment

Pushing to `master` runs `.github/workflows/deploy.yml`, which builds the site and publishes it to the `gh-pages` branch.
In the repository settings, **Pages → Build and deployment** must be set to **Deploy from a branch → `gh-pages`**, and
**Actions → General → Workflow permissions** must be **Read and write**.

See the [al-folio documentation](https://github.com/alshedivat/al-folio/tree/main/docs) for customization options.
