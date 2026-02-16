<template>
  <main class="main-content genre-select-view">
    <div class="genre-container">
      <div class="genre-card">
        <div class="genre-icon">🎭</div>
        <h2 class="genre-title">Select Your Favorite Genres</h2>
        <p class="genre-subtitle">
          Choose at least 3 genres to personalize your recommendations
        </p>

        <div class="genre-input-wrapper">
          <input v-model="searchQuery" type="text" class="genre-search-input" placeholder="Search genres..."
            :disabled="loading" @input="filterGenres" />
        </div>

        <div class="genre-list">
          <div v-for="genre in filteredGenres" :key="genre.genre_id" class="genre-item">
            <label class="genre-checkbox-label">
              <input type="checkbox" :value="genre.genre_id" v-model="selectedGenreIds" class="genre-checkbox"
                :disabled="loading" />
              <span class="genre-name">{{ genre.name }}</span>
            </label>
          </div>
        </div>

        <div v-if="filteredGenres.length === 0" class="no-genres">
          No genres found matching "{{ searchQuery }}"
        </div>

        <div class="genre-validation">
          <span v-if="selectedGenreIds.length < 3" class="validation-message error">
            ✗ Select at least 3 genres
          </span>
          <span v-else class="validation-message success">
            ✓ {{ selectedGenreIds.length }} genres selected
          </span>
        </div>

        <button type="button" class="btn-primary btn-save-genres" @click="handleSaveGenres"
          :disabled="loading || selectedGenreIds.length < 3">
          <span v-if="!loading">Save Preferences</span>
          <span v-else class="button-spinner"></span>
        </button>

        <div v-if="error" class="error-message">{{ error }}</div>
      </div>
    </div>
  </main>
</template>

<script>
import { getGenres, saveUserGenres } from '../services/api.js'

export default {
  emits: ['genres-saved'],

  data() {
    return {
      genres: [],
      filteredGenres: [],
      selectedGenreIds: [],
      searchQuery: '',
      loading: false,
      error: ''
    }
  },

  methods: {
    async loadGenres() {
      try {
        const data = await getGenres()
        this.genres = data.genres || data
        this.filteredGenres = this.genres
      } catch (err) {
        this.error = 'Failed to load genres. Please try again.'
        console.error('Error loading genres:', err)
      }
    },

    filterGenres() {
      if (!this.searchQuery.trim()) {
        this.filteredGenres = this.genres
        return
      }
      const query = this.searchQuery.toLowerCase()
      this.filteredGenres = this.genres.filter(g =>
        g.name.toLowerCase().includes(query)
      )
    },

    async handleSaveGenres() {
      if (this.selectedGenreIds.length < 3) {
        this.error = 'Please select at least 3 genres'
        return
      }

      this.loading = true
      this.error = ''

      try {
        const userId = sessionStorage.getItem('userId')
        await saveUserGenres({
          user_id: userId,
          genre_ids: this.selectedGenreIds
        })
        this.$emit('genres-saved')
      } catch (err) {
        this.error = err.message || 'Failed to save genres. Please try again.'
        console.error('Error saving genres:', err)
      } finally {
        this.loading = false
      }
    }
  },

  mounted() {
    this.loadGenres()
  }
}
</script>

<style scoped>
.genre-select-view {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
}

.genre-container {
  width: 100%;
  max-width: 600px;
  padding: 20px;
  display: flex;
  justify-content: center;
}

.genre-card {
  background: var(--bg-primary);
  border-radius: 16px;
  padding: 40px;
  box-shadow: var(--shadow-lg);
  animation: slideUp 0.6s ease-out;
  width: 100%;
}

.genre-icon {
  font-size: 64px;
  text-align: center;
  margin-bottom: 20px;
  display: block;
}

.genre-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  text-align: center;
  margin-bottom: 8px;
}

.genre-subtitle {
  font-size: 14px;
  color: var(--text-tertiary);
  text-align: center;
  margin-bottom: 24px;
}

.genre-input-wrapper {
  margin-bottom: 20px;
}

.genre-search-input {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid var(--border-color);
  border-radius: 8px;
  font-size: 14px;
  color: var(--text-primary);
  transition: all 0.3s ease;
  background: var(--bg-secondary);
}

.genre-search-input:focus {
  outline: none;
  border-color: var(--primary-color);
  background: var(--bg-primary);
  box-shadow: 0 0 0 3px rgba(167, 199, 231, 0.1);
}

.genre-list {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  max-height: 300px;
  overflow-y: auto;
  margin-bottom: 20px;
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: 8px;
}

.genre-item {
  display: flex;
  align-items: center;
}

.genre-checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
}

.genre-checkbox {
  width: 18px;
  height: 18px;
  cursor: pointer;
  accent-color: var(--primary-color);
}

.genre-checkbox:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.genre-name {
  font-size: 14px;
  color: var(--text-primary);
  font-weight: 500;
}

.no-genres {
  text-align: center;
  color: var(--text-tertiary);
  font-size: 14px;
  padding: 20px;
}

.genre-validation {
  text-align: center;
  margin-bottom: 20px;
  font-size: 14px;
  font-weight: 600;
}

.validation-message {
  padding: 8px 12px;
  border-radius: 6px;
  display: inline-block;
}

.validation-message.error {
  color: var(--error-color);
  background: rgba(239, 68, 68, 0.1);
}

.validation-message.success {
  color: var(--success-color);
  background: rgba(34, 197, 94, 0.1);
}

.btn-save-genres {
  width: 100%;
  padding: 12px 16px;
  font-size: 16px;
  font-weight: 600;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  background: var(--bg-secondary);
  color: var(--primary-purple-dark);
  margin-bottom: 16px;
}

.btn-save-genres:hover:not(:disabled) {
  background: var(--primary-dark);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.btn-save-genres:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.button-spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.error-message {
  font-size: 13px;
  color: var(--error-color);
  text-align: center;
  padding: 8px;
  background: rgba(239, 68, 68, 0.1);
  border-radius: 6px;
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

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
