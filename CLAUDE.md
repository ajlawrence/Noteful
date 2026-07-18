# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Noteful is a small Ruby on Rails app for creating and saving personal notes. Users sign up / sign in via Devise; each user owns their notes.

- Ruby 2.1.2, Rails 4.2.0.beta4 (see `Gemfile` — do not assume modern Rails APIs)
- Devise for authentication (pinned to the `lm-rails-4-2` branch of a git fork)
- SQLite in development/test, PostgreSQL in production (`rails_12factor`, Heroku-style)
- Asset pipeline with CoffeeScript, SCSS, jQuery, Turbolinks; Kube CSS framework in `app/assets/stylesheets/kube.css`

## Common commands

```bash
bin/setup            # install deps, prepare db (runs bundle, db setup)
bundle install       # install gems
bin/rake db:setup    # create db, load schema, seed
bin/rails server     # run dev server
bin/rake test        # run the Minitest suite (test/)
bin/rails console    # app console
```

Run a single test file with `bin/rake test TEST=test/models/note_test.rb`.

## Architecture

Classic Rails MVC, no API layer, no background jobs.

- **Models** (`app/models/`)
  - `Note` — `belongs_to :user`; validates presence of `title` and `content`; `sorted` scope orders by `created_at DESC`; `#preview` returns the first ~100 chars of content with a "…(readmore)" suffix.
  - `User` — Devise model (`database_authenticatable, registerable, recoverable, rememberable, trackable, validatable`); `has_many :notes`.
- **Controllers** (`app/controllers/`)
  - `NotesController` — full CRUD, gated by `before_action :authenticate_user!`. Strong params permit only `:title` and `:content`.
  - `HomeController#index` — public landing page, also the root route.
- **Routes** (`config/routes.rb`) — `root 'home#index'`, `devise_for :users`, `resources :notes`, plus `get 'home/index'`.
- **Views** — ERB in `app/views/notes` (with a shared `_form` partial) and customized Devise views in `app/views/devise`.
- **Schema** (`db/schema.rb`) — `notes(title, content, user_id, timestamps)` and the standard Devise `users` table.

## Gotchas

- `app/models/notes2.rb` is a leftover duplicate that re-opens the `Note` class (no `#preview`). Prefer editing `app/models/note.rb`; be aware both files define `Note`.
- All `NotesController` actions must scope through `current_user.notes` (e.g. `current_user.notes.find(params[:id])`), never unscoped `Note.find` — notes are per-user and unscoped lookups let one user access another's notes.
- The Rails and gem versions are old betas; generators and newer-Rails idioms (e.g. `belongs_to` required-by-default, `form_with`) do not apply here.
