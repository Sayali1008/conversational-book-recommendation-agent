# ORM Usage Guide

This document explains how to use the new ORM (Object-Relational Mapping) layer for database operations instead of writing raw SQL.

## Overview

The ORM provides:
- **Type-safe models** for all database entities
- **Repository pattern** for data access operations
- **Query builder** for complex queries
- **Automatic parameterization** (SQL injection protection)
- **Clean, readable code** instead of raw SQL strings

## Models

### Available Models
All models are dataclasses in `db/models.py`:

- `User` - User accounts
- `Genre` - Book genres
- `Author` - Book authors
- `Book` - Books
- `UserGenre` - User genre preferences (relationship)
- `BookAuthor` - Book-author associations (relationship)
- `BookGenre` - Book-genre associations (relationship)
- `Interaction` - User interactions (likes/dislikes)
- `Rating` - User ratings

## Repositories

### UserRepository
```python
from db.models import User, UserRepository

# Create a user
user = User(user_id="user123", name="John Doe")
UserRepository.create(user)

# Find user by ID
user = UserRepository.find_by_id("user123")

# Check if user exists
exists = UserRepository.exists("user123")

# Update user
user.login_attempt = 1
UserRepository.update(user)
```

### GenreRepository
```python
from db.models import Genre, GenreRepository

# Create genre
genre = Genre(genre_id=None, name="Fiction")
genre_id = GenreRepository.create(genre)

# Find by ID or name
genre = GenreRepository.find_by_id(1)
genre = GenreRepository.find_by_name("Fiction")

# Get all genres
genres = GenreRepository.find_all()
```

### AuthorRepository
```python
from db.models import Author, AuthorRepository

# Create author
author = Author(author_id=None, name="Jane Austen")
author_id = AuthorRepository.create(author)

# Find by ID or name
author = AuthorRepository.find_by_id(1)
author = AuthorRepository.find_by_name("Jane Austen")

# Search authors
results = AuthorRepository.search("jane")

# Get all authors
authors = AuthorRepository.find_all()
```

### BookRepository
```python
from db.models import Book, BookRepository
from datetime import datetime

# Create book
book = Book(
    book_id=1001,
    title="Pride and Prejudice",
    description="A novel about...",
    infolink="https://...",
    created_at=datetime.now().isoformat()
)
BookRepository.create(book)

# Find book by ID
book = BookRepository.find_by_id(1001)

# Get next available book_id
max_id = BookRepository.get_max_id()
next_id = max_id + 1
```

### UserGenreRepository
```python
from db.models import UserGenre, UserGenreRepository
from datetime import datetime

# Add genre preference for user
user_genre = UserGenre(
    user_id="user123",
    genre_id=1,
    created_at=datetime.now().isoformat()
)
UserGenreRepository.create(user_genre)

# Get user's genres
genres = UserGenreRepository.find_by_user("user123")
# Returns: List[Genre] with genre_id and name

# Check if user has genres
has_genres = UserGenreRepository.user_has_genres("user123")

# Delete user's genre preferences
UserGenreRepository.delete_by_user("user123")
```

### BookAuthorRepository
```python
from db.models import BookAuthor, BookAuthorRepository

# Create book-author association
assoc = BookAuthor(book_id=1001, author_id=5)
BookAuthorRepository.create(assoc)

# Get all authors for a book
authors = BookAuthorRepository.find_authors_by_book(1001)
# Returns: List[Author] with author_id and name
```

### BookGenreRepository
```python
from db.models import BookGenre, BookGenreRepository

# Create book-genre association
assoc = BookGenre(book_id=1001, genre_id=3)
BookGenreRepository.create(assoc)

# Get all genres for a book
genres = BookGenreRepository.find_genres_by_book(1001)
# Returns: List[Genre] with genre_id and name

# Find popular books by genre
popular_books = BookGenreRepository.find_popular_books_by_genre(
    genre_id=3,
    limit=5
)
# Returns: List[Dict] with book_id and rating_count
```

### InteractionRepository
```python
from db.models import Interaction, InteractionRepository
from datetime import datetime

# Create interaction
interaction = Interaction(
    user_id="user123",
    book_id=1001,
    action="like",
    confidence=0.95,
    ts=datetime.now().isoformat()
)
InteractionRepository.create(interaction)

# Get user's interactions
interactions = InteractionRepository.find_by_user("user123")

# Get specific action interactions
likes = InteractionRepository.find_by_user("user123", actions=["like"])

# Get limited interactions
recent = InteractionRepository.find_by_user("user123", limit=10)

# Combine filters
recent_likes = InteractionRepository.find_by_user(
    "user123",
    actions=["like"],
    limit=10
)
```

## Query Builder

For more complex queries, use the `Query` class:

```python
from db.models import Query

# Simple SELECT
results = (
    Query("books")
    .select(["title", "author", "rating"])
    .where("rating > ?", 4.0)
    .order_by("rating", "DESC")
    .limit(10)
    .execute()
)

# Get single result
result = (
    Query("books")
    .where("book_id = ?", 101)
    .execute_one()
)

# Get scalar value
count = (
    Query("books")
    .select(["COUNT(*) as count"])
    .execute_scalar()
)
```

## Migration Guide: From Raw SQL to ORM

### Before (Raw SQL)
```python
def get_user_genres(user_id):
    db = get_db()
    conn = db.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT g.genre_id, g.name FROM user_genres ug "
            "JOIN genres g ON ug.genre_id = g.genre_id "
            "WHERE ug.user_id = ?",
            (user_id,)
        )
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()
```

### After (ORM)
```python
from db.models import UserGenreRepository

def get_user_genres(user_id):
    return UserGenreRepository.find_by_user(user_id)
```

## Benefits

1. **Type Safety**: Models are typed, catching errors early
2. **SQL Injection Protection**: All parameters are automatically sanitized
3. **Less Code**: Reduce boilerplate database operations
4. **Readability**: Intent is clear without reading SQL
5. **Maintainability**: Changes to schema affect one place
6. **Testability**: Easy to mock repositories for testing
7. **Consistency**: All database access follows same pattern

## When to Use Query Builder vs Repository

- **Repository**: Use for standard operations (find, create, search)
- **Query Builder**: Use for custom/complex queries not covered by repositories

Example:
```python
# Use repository - simple
genres = GenreRepository.find_all()

# Use Query Builder - complex
results = (
    Query("books")
    .select(["b.book_id", "b.title", "COUNT(r.user_id) as rating_count"])
    .join("JOIN book_genres bg ON b.book_id = bg.book_id")
    .join("LEFT JOIN ratings r ON b.book_id = r.book_id")
    .where("bg.genre_id = ?", genre_id)
    .where("b.created_at > ?", cutoff_date)
    .order_by("rating_count", "DESC")
    .limit(10)
    .execute()
)
```

## Future Enhancements

- Batch operations (insert_many, update_many)
- Eager loading for relationships
- Lazy loading optimization
- Query caching
- Transaction support
- Migration framework
