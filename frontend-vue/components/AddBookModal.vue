<template>
  <div class="modal-backdrop" @click="close">
    <div class="modal-content" @click.stop>
      <button class="modal-close" @click="close">✕</button>
      
      <h2 class="modal-title">📚 Add New Book</h2>

      <form @submit.prevent="handleSubmit" class="form">
        <!-- Title -->
        <div class="form-group">
          <label for="title" class="form-label">Title *</label>
          <input
            id="title"
            v-model="form.title"
            type="text"
            class="form-input"
            placeholder="Enter book title"
            required
            :disabled="loading"
          />
        </div>

        <!-- Description -->
        <div class="form-group">
          <label for="description" class="form-label">Description *</label>
          <textarea
            id="description"
            v-model="form.description"
            class="form-textarea"
            placeholder="Enter book description"
            rows="4"
            required
            :disabled="loading"
          ></textarea>
          <p class="form-help">{{ form.description.length }} characters</p>
        </div>

        <!-- Authors -->
        <div class="form-group">
          <label class="form-label">Authors *</label>
          <div class="multi-select-wrapper">
            <input
              v-model="authorSearch"
              type="text"
              class="form-input"
              placeholder="Search or type author name..."
              @input="filterAuthors"
              :disabled="loading"
            />
            <div v-if="authorSearch && filteredAuthors.length" class="dropdown">
              <div
                v-for="author in filteredAuthors"
                :key="author.author_id"
                class="dropdown-item"
                @click="selectAuthor(author)"
              >
                {{ author.name }}
              </div>
            </div>
            <div v-if="authorSearch && !filteredAuthors.length && !creatingAuthor" class="dropdown">
              <div class="dropdown-item add-new" @click="startCreateAuthor">
                ➕ Create new: "{{ authorSearch }}"
              </div>
            </div>
          </div>

          <div v-if="creatingAuthor" class="inline-create">
            <input
              v-model="newAuthorName"
              type="text"
              class="form-input"
              placeholder="Author name"
              @keyup.enter="confirmCreateAuthor"
              @keyup.esc="cancelCreateAuthor"
              autofocus
            />
            <button type="button" class="btn-mini btn-success" @click="confirmCreateAuthor">✓</button>
            <button type="button" class="btn-mini btn-cancel" @click="cancelCreateAuthor">✕</button>
          </div>

          <div v-if="form.authors.length" class="selected-items">
            <div v-for="authorId in form.authors" :key="authorId" class="selected-item">
              {{ getAuthorName(authorId) }}
              <button type="button" class="remove-btn" @click="removeAuthor(authorId)" :disabled="loading">
                ✕
              </button>
            </div>
          </div>
        </div>

        <!-- Genres -->
        <div class="form-group">
          <label class="form-label">Genres *</label>
          <div class="multi-select-wrapper">
            <input
              v-model="genreSearch"
              type="text"
              class="form-input"
              placeholder="Search or type genre..."
              @input="filterGenres"
              :disabled="loading"
            />
            <div v-if="genreSearch && filteredGenres.length" class="dropdown">
              <div
                v-for="genre in filteredGenres"
                :key="genre.genre_id"
                class="dropdown-item"
                @click="selectGenre(genre)"
              >
                {{ genre.name }}
              </div>
            </div>
            <div v-if="genreSearch && !filteredGenres.length && !creatingGenre" class="dropdown">
              <div class="dropdown-item add-new" @click="startCreateGenre">
                ➕ Create new: "{{ genreSearch }}"
              </div>
            </div>
          </div>

          <div v-if="creatingGenre" class="inline-create">
            <input
              v-model="newGenreName"
              type="text"
              class="form-input"
              placeholder="Genre name"
              @keyup.enter="confirmCreateGenre"
              @keyup.esc="cancelCreateGenre"
              autofocus
            />
            <button type="button" class="btn-mini btn-success" @click="confirmCreateGenre">✓</button>
            <button type="button" class="btn-mini btn-cancel" @click="cancelCreateGenre">✕</button>
          </div>

          <div v-if="form.genres.length" class="selected-items">
            <div v-for="genreId in form.genres" :key="genreId" class="selected-item">
              {{ getGenreName(genreId) }}
              <button type="button" class="remove-btn" @click="removeGenre(genreId)" :disabled="loading">
                ✕
              </button>
            </div>
          </div>
        </div>

        <!-- Info Link -->
        <div class="form-group">
          <label for="infolink" class="form-label">Info Link (optional)</label>
          <input
            id="infolink"
            v-model="form.infolink"
            type="url"
            class="form-input"
            placeholder="https://example.com/book"
            :disabled="loading"
          />
        </div>

        <!-- Form Actions -->
        <div class="form-actions">
          <button type="button" class="btn-cancel" @click="close" :disabled="loading">
            Cancel
          </button>
          <button type="submit" class="btn-save" :disabled="loading || !isFormValid">
            <span v-if="!loading">Save Book</span>
            <span v-else>Saving...</span>
          </button>
        </div>

        <p v-if="error" class="error-message">{{ error }}</p>

        <!-- Duplicate warning -->
        <div v-if="potentialDuplicate" class="duplicate-warning">
          <div class="warning-card">
            <div class="warning-icon">⚠️</div>
            <h3>Book May Already Exist</h3>
            <p>We found {{ duplicateBooks.length }} book(s) with this title and similar author(s):</p>
            <!-- <div class="duplicate-list">
              <div v-for="book in duplicateBooks" :key="book.book_id" class="duplicate-item">
                <div class="duplicate-title">{{ book.title }}</div>
                <div class="duplicate-authors">by {{ book.authors }}</div>
              </div>
            </div> -->
            <div class="warning-actions">
              <button type="button" class="btn-cancel" @click="cancelDuplicate">
                Cancel
              </button>
              <button type="button" class="btn-proceed" @click="proceedWithDuplicate">
                Add Anyway
              </button>
            </div>
          </div>
        </div>
      </form>
    </div>
  </div>
</template>

<script>
import { addBook, getAuthors, createAuthor, getGenres, createGenre, searchBooks } from '../services/api.js'

export default {
  emits: ['book-added', 'close'],

  data() {
    return {
      form: {
        title: '',
        description: '',
        authors: [],
        genres: [],
        infolink: ''
      },
      authors: [],
      authorSearch: '',
      filteredAuthors: [],
      creatingAuthor: false,
      newAuthorName: '',
      genres: [],
      genreSearch: '',
      filteredGenres: [],
      creatingGenre: false,
      newGenreName: '',
      loading: false,
      error: '',
      potentialDuplicate: null,
      confirmingDuplicate: false,
      duplicateBooks: []
    }
  },

  computed: {
    isFormValid() {
      return (
        this.form.title.trim().length > 0 &&
        this.form.description.trim().length > 0 &&
        this.form.authors.length > 0 &&
        this.form.genres.length > 0
      )
    }
  },

  methods: {
    async loadAuthorsAndGenres() {
      try {
        const authorsData = await getAuthors()
        this.authors = authorsData.authors || authorsData
        const genresData = await getGenres()
        this.genres = genresData.genres || genresData
      } catch (error) {
        console.error('Error loading data:', error)
      }
    },

    filterAuthors() {
      if (!this.authorSearch.trim()) {
        this.filteredAuthors = []
        return
      }
      const query = this.authorSearch.toLowerCase()
      this.filteredAuthors = this.authors.filter(
        a => a.name.toLowerCase().includes(query) && !this.form.authors.includes(a.author_id)
      )
    },

    selectAuthor(author) {
      if (!this.form.authors.includes(author.author_id)) {
        this.form.authors.push(author.author_id)
      }
      this.authorSearch = ''
      this.filteredAuthors = []
    },

    removeAuthor(authorId) {
      this.form.authors = this.form.authors.filter(id => id !== authorId)
    },

    getAuthorName(authorId) {
      const author = this.authors.find(a => a.author_id === authorId)
      return author ? author.name : 'Unknown'
    },

    startCreateAuthor() {
      this.creatingAuthor = true
      this.newAuthorName = this.authorSearch
    },

    async confirmCreateAuthor() {
      if (!this.newAuthorName.trim()) return
      try {
        const newAuthor = await createAuthor(this.newAuthorName)
        this.authors.push(newAuthor)
        this.form.authors.push(newAuthor.author_id)
        this.authorSearch = ''
        this.creatingAuthor = false
        this.newAuthorName = ''
      } catch (error) {
        console.error('Error creating author:', error)
      }
    },

    cancelCreateAuthor() {
      this.creatingAuthor = false
      this.newAuthorName = ''
    },

    filterGenres() {
      if (!this.genreSearch.trim()) {
        this.filteredGenres = []
        return
      }
      const query = this.genreSearch.toLowerCase()
      this.filteredGenres = this.genres.filter(
        g => g.name.toLowerCase().includes(query) && !this.form.genres.includes(g.genre_id)
      )
    },

    selectGenre(genre) {
      if (!this.form.genres.includes(genre.genre_id)) {
        this.form.genres.push(genre.genre_id)
      }
      this.genreSearch = ''
      this.filteredGenres = []
    },

    removeGenre(genreId) {
      this.form.genres = this.form.genres.filter(id => id !== genreId)
    },

    getGenreName(genreId) {
      const genre = this.genres.find(g => g.genre_id === genreId)
      return genre ? genre.name : 'Unknown'
    },

    startCreateGenre() {
      this.creatingGenre = true
      this.newGenreName = this.genreSearch
    },

    async confirmCreateGenre() {
      if (!this.newGenreName.trim()) return
      try {
        const newGenre = await createGenre(this.newGenreName)
        this.genres.push(newGenre)
        this.form.genres.push(newGenre.genre_id)
        this.genreSearch = ''
        this.creatingGenre = false
        this.newGenreName = ''
      } catch (error) {
        console.error('Error creating genre:', error)
      }
    },

    cancelCreateGenre() {
      this.creatingGenre = false
      this.newGenreName = ''
    },

    async handleSubmit() {
      this.error = ''
      this.potentialDuplicate = null
      this.loading = true

      try {
        // Check for potential duplicates before submitting
        if (!this.confirmingDuplicate) {
          const isDuplicate = await this.checkForDuplicate()
          if (isDuplicate) {
            this.loading = false
            this.potentialDuplicate = isDuplicate
            this.confirmingDuplicate = true
            return
          }
        }

        const payload = {
          title: this.form.title.trim(),
          description: this.form.description.trim(),
          authors: this.form.authors,
          genres: this.form.genres,
          infolink: this.form.infolink.trim() || null
        }
        console.log('Submitting payload:', payload)
        console.log('Authors array:', this.form.authors)
        console.log('Genres array:', this.form.genres)
        
        const response = await addBook(payload)
        this.$emit('book-added', response)
        this.confirmingDuplicate = false
      } catch (error) {
        this.error = error.message || 'Failed to add book'
        console.error('Error adding book:', error)
        console.error('Full form data:', this.form)
        console.log('Full error object:', error)
      } finally {
        this.loading = false
      }
    },

    async checkForDuplicate() {
      try {
        // Search for books with same title and any of the selected authors using author IDs
        console.log('Checking duplicates - Title:', this.form.title.trim())
        console.log('Checking duplicates - Author IDs:', this.form.authors)
        
        const result = await searchBooks(this.form.title.trim(), this.form.authors)
        console.log('Duplicate check result:', result)
        
        if (result.books && result.books.length > 0) {
          this.duplicateBooks = result.books
          return true
        }
        return false
      } catch (error) {
        console.error('Error checking for duplicate:', error)
        return false
      }
    },

    proceedWithDuplicate() {
      this.confirmingDuplicate = true
      this.handleSubmit()
    },

    cancelDuplicate() {
      this.potentialDuplicate = null
      this.confirmingDuplicate = false
    },

    close() {
      this.$emit('close')
    }
  },

  mounted() {
    this.loadAuthorsAndGenres()
    document.body.style.overflow = 'hidden'
  },

  beforeUnmount() {
    document.body.style.overflow = 'auto'
  }
}
</script>

<style scoped>
.modal-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(45, 27, 61, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  padding: 20px;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.modal-content {
  background-color: var(--bg-primary);
  border-radius: 16px;
  box-shadow: var(--shadow-xl);
  max-width: 500px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  position: relative;
  animation: slideUp 0.3s ease;
  padding: 40px;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.modal-close {
  position: absolute;
  top: 16px;
  right: 16px;
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: var(--text-secondary);
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: all 0.2s ease;
}

.modal-close:hover {
  background-color: var(--bg-secondary);
  color: var(--text-primary);
}

.modal-title {
  font-size: 22px;
  font-weight: 700;
  color: var(--primary-pink-dark);
  margin: 0 0 24px 0;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-label {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--primary-pink-dark);
}

.form-input,
.form-textarea {
  padding: 12px;
  border: 2px solid var(--border-color);
  border-radius: 8px;
  font-size: 14px;
  font-family: inherit;
  color: var(--text-primary);
  background-color: var(--bg-primary);
  transition: all 0.2s ease;
}

.form-input:focus,
.form-textarea:focus {
  outline: none;
  border-color: var(--primary-pink-dark);
  box-shadow: 0 0 0 3px rgba(243, 154, 200, 0.1);
}

.form-input:disabled,
.form-textarea:disabled {
  background-color: var(--bg-secondary);
  cursor: not-allowed;
  opacity: 0.6;
}

.form-help {
  font-size: 12px;
  color: var(--text-tertiary);
  margin: 0;
}

.multi-select-wrapper {
  position: relative;
}

.dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-top: none;
  border-radius: 0 0 8px 8px;
  max-height: 200px;
  overflow-y: auto;
  z-index: 100;
}

.dropdown-item {
  padding: 10px 12px;
  font-size: 14px;
  color: var(--text-primary);
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.dropdown-item:hover {
  background-color: var(--bg-secondary);
}

.dropdown-item.add-new {
  color: var(--primary-pink-dark);
  font-weight: 600;
}

.inline-create {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}

.inline-create .form-input {
  flex: 1;
}

.btn-mini {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-success {
  background-color: var(--success-color);
  color: white;
}

.btn-success:hover {
  background-color: #95D8C1;
}

.btn-cancel {
  background-color: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
}

.btn-cancel:hover {
  background-color: var(--bg-secondary);
}

.selected-items {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.selected-item {
  display: flex;
  align-items: center;
  gap: 8px;
  background-color: var(--bg-secondary);
  color: var(--text-primary);
  padding: 8px 12px;
  border-radius: 20px;
  font-size: 13px;
}

.remove-btn {
  background: none;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 14px;
  padding: 0;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color 0.2s ease;
}

.remove-btn:hover:not(:disabled) {
  color: var(--error-color);
}

.remove-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.form-actions {
  display: flex;
  gap: 12px;
  margin-top: 24px;
  justify-content: flex-end;
}

.btn-cancel,
.btn-save {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-cancel {
  background-color: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
}

.btn-cancel:hover:not(:disabled) {
  background-color: var(--bg-secondary);
}

.btn-save {
  background-color: var(--primary-pink);
  color: white;
  border: none;
}

.btn-save:hover:not(:disabled) {
  background-color: var(--primary-pink-dark);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.btn-save:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error-message {
  color: var(--error-color);
  font-size: 13px;
  margin-top: 16px;
  padding: 12px;
  background-color: rgba(248, 168, 168, 0.1);
  border-radius: 6px;
}

.duplicate-warning {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(45, 27, 61, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3000;
}

.warning-card {
  background-color: var(--bg-primary);
  border-radius: 12px;
  padding: 40px;
  max-width: 400px;
  box-shadow: var(--shadow-xl);
  text-align: center;
  animation: slideUp 0.3s ease;
}

.warning-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.warning-card h3 {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 12px 0;
}

.warning-card p {
  font-size: 14px;
  color: var(--text-secondary);
  margin: 0 0 24px 0;
  line-height: 1.5;
}

.warning-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
}

.btn-proceed {
  background-color: var(--primary-pink);
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-proceed:hover {
  background-color: var(--primary-pink-dark);
  transform: translateY(-2px);
}
</style>
