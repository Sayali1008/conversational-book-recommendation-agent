# Cold-Start Strategy: Genre-Based Personalization & Incremental Learning

## Overview
This document outlines the implementation strategy for handling **cold users** (no interaction history) and **cold books** (newly added without CF training) in the recommendation system, including user authentication, preference capture, and dynamic book management.

---

## 1. Problem Statement

### Current State
- **Cold users** (unknown `user_id` or no ratings): Fall back to content-based recommendations using embeddings only
- **Cold books** (not in CF training data): Available for content-based recommendations but not CF scoring
- **Cold starts** lack personalization: All cold users see similar recommendations (catalog mean embedding)
- **Data stored in files**: Books, users, ratings in CSV/FTR files, not easily updatable

### Goals
1. **Personalize cold-user experience** by capturing initial genre preferences upfront
2. **Boost cold books** that match user preferences so newly added books surface early
3. **Enable on-demand retraining** to refresh CF models with new books without waiting
4. **Support dynamic user and book management** via database-backed forms
5. **Minimize code changes** by leveraging existing hybrid scoring infrastructure

---

## 2. User Registration & Authentication Flow

### Overview
Users authenticate without passwords. First-time users create accounts; returning users log in and optionally update preferences.

### User Registration (Create Account)

**Form Inputs:**
1. **User Name** (required)
   - Display name, stored in `users.name`
   - No uniqueness constraint
   - Example: "Alice Smith"

2. **User ID** (required, unique)
   - System identifier, stored in `users.user_id`
   - Must be unique across system
   - Real-time validation: check against existing users
   - Example: "alice_smith_001"

**Account Creation Steps:**
1. User enters name and user ID
2. Backend validates user ID uniqueness
   - If exists: Show error "User ID already taken, try another"
   - If available: Proceed to step 3
3. Create user record in `users` table:
   ```sql
   INSERT INTO users (user_id, name, login_attempt, created_at, last_login)
   VALUES ('alice_smith_001', 'Alice Smith', 0, NOW(), NOW())
   ```
4. Redirect to **Genre Selection screen**

### User Login

**Form Inputs:**
1. **User ID** (required)
   - User enters their ID
   
**Login Steps:**
1. Backend looks up user in `users` table
2. If found:
   - Increment `login_attempt`
   - Update `last_login` timestamp
   - Check if user has genre preferences
     - If YES: Proceed to recommendations
     - If NO: Redirect to **Genre Selection screen** (first login)
3. If not found:
   - Show error: "User ID not found. Please create an account first."
   - Offer link to Create Account screen

### Genre Preference Selection

**Form Inputs:**
1. **Genre Multi-Select Dropdown** (required)
   - Source: Top 50 genres from catalog (stored in `genres` table)
   - Minimum selection: 3 genres (lower bound for profile diversity)
   - Recommended selection: 5 genres (initial guidance)
   - Maximum selection: 10 genres (avoid over-specification)
   - Auto-complete: Typing filters genre list

**Genre Selection Steps:**
1. User selects genres from dropdown
2. Validation: Enforce minimum 3 genres
   - If < 3: Show helper text "Select at least 3 genres to continue"
3. User clicks "Save Preferences"
4. Backend stores in `user_genres` table:
   ```sql
   INSERT INTO user_genres (user_id, genre_id, created_at)
   VALUES ('alice_smith_001', 1, NOW()),  -- e.g., 'mystery' genre_id=1
          ('alice_smith_001', 5, NOW()),  -- e.g., 'fiction' genre_id=5
          ...
   ```
5. Build user profile from embeddings of popular books in selected genres
6. Redirect to **Recommendations screen**

**Updating Preferences:**
- Future enhancement: Add "Preferences" or "Settings" page
- Allows users to change genres anytime
- Deletes old preferences, inserts new ones

---

## 3. Cold Books Management

### Overview
Admins/users can add new books to the catalog. Books are stored in the database and become immediately available for content-based recommendations. CF recommendations require model retraining.

### Book Addition Form

**Form Inputs:**
1. **Title** (required, text input)
   - Example: "The Mystery of the Silent Library"

2. **Authors** (required, searchable multi-select)
   - Source: `authors` table (searchable dropdown with auto-complete)
   - Typing triggers search: "Type author name..."
   - Option to add new author if not found
   - Multiple authors allowed
   - Example: ["Author One", "Author Two"]

3. **Description** (required, textarea)
   - Minimum length: `DATA_PREPROCESSING['min_desc_length']` = 10 characters
   - Validation: Real-time character count + validation message
   - Help text: "Minimum 10 characters required"
   - Example: Multi-paragraph description of plot

4. **Genres** (required, searchable multi-select)
   - Source: `genres` table (searchable dropdown with auto-complete)
   - Typing triggers search
   - Option to add new genre if not found
   - Multiple genres allowed
   - Example: ["mystery", "fiction", "thriller"]

5. **Info Link** (optional, URL input)
   - Example: "https://www.goodreads.com/book/..."
   - Validation: Valid URL format

### Book Addition Steps

1. **Form Validation** (frontend):
   - Title: non-empty
   - Authors: at least one selected
   - Description: length ≥ 10 characters
   - Genres: at least one selected
   - Info Link: valid URL format (if provided)

2. **Author/Genre Lookup** (backend):
   - Check if each author exists in `authors` table
   - If not found: Create new author record
   - Check if each genre exists in `genres` table
   - If not found: Create new genre record

3. **Create Book Record** (backend):
   ```sql
   INSERT INTO books (title, description, infolink, created_at)
   VALUES ('The Mystery of the Silent Library', 'A gripping tale...', 'https://...', NOW())
   RETURNING book_id;
   ```

4. **Link Authors & Genres** (backend):
   ```sql
   -- Link to authors
   INSERT INTO book_authors (book_id, author_id) VALUES (1001, 42), (1001, 15);
   
   -- Link to genres
   INSERT INTO book_genres (book_id, genre_id) VALUES (1001, 3), (1001, 7), (1001, 12);
   ```

5. **Generate Embedding** (immediate or deferred):
   - **Immediate option**: Call embedding model on-the-fly
     - Pros: Book becomes recommendable immediately
     - Cons: Latency in form submission
   - **Deferred option**: Queue for batch embedding later
     - Pros: Fast form submission
     - Cons: Book not immediately recommendable
   - **Recommendation**: Start with **Deferred** (simpler), move to **Immediate** with async processing later

6. **Book Availability**:
   - **Content-based recommendations**: Available immediately (if embedding generated)
   - **Collaborative filtering**: Available only after model retraining
   - **Search & discovery**: Available immediately in UI

### New Author/Genre Creation

When user types an author/genre not found in dropdown:

1. **Show "Create New" Option**
   - Example: "mystery_romance (create new)"
   
2. **Confirm Creation** (optional confirmation modal)
   - "Add new genre: 'mystery_romance'?"
   
3. **Insert Record**:
   ```sql
   INSERT INTO genres (name) VALUES ('mystery_romance');
   ```

4. **Auto-select in Form**
   - Newly created genre selected in multi-select

---

## 4. Database Schema

### Users Table
```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    login_attempt INT DEFAULT 0,
    last_login TIMESTAMP
);
```

### Genres Table
```sql
CREATE TABLE genres (
    genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);
```

### Authors Table
```sql
CREATE TABLE authors (
    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);
```

### Books Table (migrated from cleaned ftr file)
```sql
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    infolink TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Book-Authors Junction Table
```sql
CREATE TABLE book_authors (
    book_id INTEGER,
    author_id INTEGER,
    PRIMARY KEY (book_id, author_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id),
    FOREIGN KEY (author_id) REFERENCES authors(author_id)
);
```

### Book-Genres Junction Table
```sql
CREATE TABLE book_genres (
    book_id INTEGER,
    genre_id INTEGER,
    PRIMARY KEY (book_id, genre_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id),
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id)
);
```

### User Genres Table
```sql
CREATE TABLE user_genres (
    user_id TEXT,
    genre_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, genre_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id)
);
```

### Interactions Table (existing, now with FK)
```sql
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    book_id INTEGER NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('like','dislike')),
    confidence REAL,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);
```

### Ratings Table (migrated from cleaned ftr file)
```sql
CREATE TABLE ratings (
    user_id TEXT,
    book_id INTEGER,
    score INT,
    confidence FLOAT,
    datetime TIMESTAMP,
    review_summary TEXT,
    review_text TEXT,
    PRIMARY KEY (user_id, book_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);
```

### Migration Steps
1. Extract top-50 genres from cleaned books catalog → insert into `genres` table
2. Extract unique authors from cleaned books catalog → insert into `authors` table
3. Load cleaned books ftr file → insert into `books` table
4. Build `book_authors` junction from catalog metadata
5. Build `book_genres` junction from catalog metadata
6. Load cleaned ratings ftr file → insert into `ratings` table
7. Extract user IDs from ratings → insert into `users` table (set login_attempt=0)

---

## 5. Filtering Previously Rated Books

### Strategy: Always Filter Rated Books

**Recommendation**: Filter **all previously rated books** from recommendations for simplicity and clarity.

**Rationale:**
- Avoids cluttering recommendations with books user has already evaluated
- Maintains clear separation: "Books you've rated" vs. "New recommendations"
- Consistent behavior across warm and cold users

**Implementation:**
- Current behavior already supports this: `filter_rated=True` in `RecommendationConfig`
- At recommendation time, check `ratings` table:
  - If user has rated book_id: exclude from final results

---

## 6. Backend: Genre-Based Profile Building

### Data Flow for Cold Users

**Step 1: Retrieve User's Preferred Genres**
- Query `user_genres` table for user_id
- Fetch genre names from `genres` table

**Step 2: Find Popular Books in Those Genres**
- Query `books` table joined with `book_genres` for books matching user's genre preferences
- Rank by popularity (average rating or count)
- Select top-K books per genre as seeds

**Step 3: Build User Profile from Genre Seeds**
- Fetch embeddings for seed books
- Create weighted average of embeddings
- Result: dense user profile vector

**Step 4: Score Candidates with Genre Boost**
- Compute base embedding similarity
- Apply multiplier boost for books matching user genres:
  ```
  boost_multiplier = 1 + (0.2 * num_matching_genres)
  final_score = base_score * boost_multiplier
  ```

**Step 5: Handle Seedless Cold Users**
- If user hasn't selected genres (shouldn't happen in new flow, but handle gracefully):
  - Fall back to **catalog mean similarity** (current behavior)
  - Or show "popular across all genres" as safe default

---

## 7. API Contract Changes

### New Endpoints

```python
@app.post("/users")
def create_user(payload: CreateUserRequest): ...

@app.post("/login")
def login(payload: LoginRequest): ...

@app.get("/genres")
def get_available_genres(): ...

@app.get("/authors")
def get_available_authors(q: Optional[str] = None): ...

@app.post("/preferred-genres")
def set_user_genres(user_id: str, payload: UserGenresRequest): ...

@app.post("/books")
def add_book(payload: AddBookRequest): ...
```

---

## 8. Implementation Roadmap

### Phase 0: Database Setup
- [x] Create database schema
- [x] Migrate data from FTR/CSV files to database
- [x] Extract top-50 genres → genres table

### Phase 1: Authentication & User Management
- [x] Create Login and Registration screens
- [x] Implement `/users` and `/login` endpoints
- [x] Test login/register flow

### Phase 2: Genre Preferences
- [x] Create Genre Selection screen
- [x] Implement `/genres` endpoint
- [x] Implement `/preferred-genres` endpoint

### Phase 3: Cold Book Management
- [ ] Create "Add Book" form
- [ ] Implement `/authors` and `/books` endpoints
- [ ] Test book addition flow

### Phase 4: Genre-Based Scoring
- [ ] Modify content-based recommender with genre boost
- [ ] Fetch user's preferred genres from DB
- [ ] Test cold user recommendations

### Phase 5: Testing & Integration
- [ ] End-to-end testing
- [ ] Verify cold book boost works
- [ ] Test login tracking

---

## 9. Success Metrics
- [ ] New users can register and select genre preferences
- [ ] Cold users see personalized recommendations
- [ ] Cold books surface in top-k for matching genres
- [ ] Login tracking works correctly
- [ ] Authors and genres can be added dynamically
```
  - Apply **multiplier boost** for cold books matching user genres:
    ```
    num_matching_genres = count(candidate_genres ∩ user_genres)
    boost_multiplier = 1 + (0.2 * num_matching_genres)
    final_score = base_score * boost_multiplier
    ```
  - Example: 2 matching genres → 1.4x multiplier
  - Rank by final_score, return top-k
```

---

## 7. API Contract Changes

### What We're NOT Changing
- Core CF algorithm (ALS remains unchanged)
- Interaction matrix construction
- Warm-user hybrid scoring (existing logic preserved)
- Swipe storage and management (SQLite unchanged)
- Pipeline stages (preprocessing, embedding, matrix building, training, eval)

### What We're Changing
1. **Frontend**: Add 1 form field (genre selector) to RecommendationForm
2. **Schemas**: Add optional `genres` field to request
3. **Content-based scoring**: Add genre-matching multiplier (3-5 lines of code)
4. **User profile building**: Support building from genres (refactor existing function slightly)
5. **API**: Add new `/genres` endpoint, modify `/recommend` signature

### Localized Impact
- Changes isolated to:
  - `server/schemas.py` (1 schema addition)
  - `server/recommendation_service.py` (genre parsing + passing)
  - `server/main.py` (new endpoint + parameter handling)
  - `recommenders/handler.py` (pass genres through)
  - `recommenders/content_based.py` (genre boost logic)
  - `recommenders/collaborative.py` (optional: genre-based profile building)
  - `frontend-vue/components/RecommendationForm.vue` (form input)
  - `frontend-vue/services/api.js` (parameter passing)

---

## 9. Expected Behavior After Implementation

### Scenario 1: New User with Genre Preferences
1. User enters: `user_id="alice"`, `genres=["mystery", "fiction"]`
2. Backend:
   - Builds profile from embeddings of popular mystery/fiction books
   - Content-based scores books, applies boost for cold books matching those genres
   - Returns 10 recommendations with new books highlighted
3. User swipes → interactions logged → swipes inform future profile refinement

### Scenario 2: New User without Preferences
1. User enters: `user_id="bob"`, `genres=[]`
2. Backend:
   - Falls back to genre+popularity scoring
   - Returns top-rated books across genres for discovery
3. After first few swipes → profile refines based on liked books

### Scenario 3: New Book in Catalog
1. Admin adds 100 new books via data upload
2. Clicks "Retrain Pipeline" button
3. Pipeline runs: preprocesses new books → generates embeddings → rebuilds matrices → retrains CF
4. New books now:
   - Eligible for CF recommendations for warm users
   - Boosted for cold users who like relevant genres
   - Stored in expanded catalog

---

## 10. Success Metrics
- [ ] Cold users with genre prefs see more relevant recommendations than before
- [ ] Cold books (newly added) surface in top-k for matching genre users
- [ ] Swipe velocity for cold users increases (more engagement)
- [ ] Code changes minimal and localized (no major refactors)
- [ ] Retraining pipeline accepts user-added books seamlessly

---

## 11. Future Enhancements (Out of Scope)
- Implicit genre signals from swipe history
- Author-based preferences
- A/B testing genre boost multiplier
- Hybrid CF+CB scoring for warm users with genre preferences
- Incremental matrix updates without full retraining
