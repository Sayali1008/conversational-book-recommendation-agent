<template>
  <div class="modal-backdrop" @click="close">
    <div class="modal-content" @click.stop>
      <button class="modal-close" @click="close">✕</button>
      
      <div class="modal-header">
        <div class="book-image">
          <div class="image-placeholder">📖</div>
        </div>
        
        <div class="book-header-info">
          <h2 class="book-title">{{ book.title }}</h2>
          <p class="book-authors">{{ formattedAuthors }}</p>
        </div>
      </div>

      <div class="modal-body">
        <div class="detail-group">
          <label class="detail-label">Match Score</label>
          <div class="detail-value score">
            <span v-if="book.score !== undefined && book.score !== null">
              ⭐ {{ (book.score * 100).toFixed(1) }}%
            </span>
            <span v-else class="text-muted">No score available</span>
          </div>
        </div>

        <div class="detail-group">
          <label class="detail-label">Genres</label>
          <div class="detail-value">
            <div v-if="book.genres && book.genres.length" class="genres-list">
              <span v-for="genre in book.genres" :key="genre.genre_id || genre" class="genre-tag">
                {{ typeof genre === 'string' ? genre : genre.name }}
              </span>
            </div>
            <p v-else class="text-muted">No genres listed</p>
          </div>
        </div>

        <div class="detail-group">
          <label class="detail-label">Description</label>
          <p class="detail-value description">{{ book.description || 'No description available' }}</p>
        </div>

        <div v-if="book.infolink" class="detail-group">
          <a :href="book.infolink" target="_blank" rel="noopener" class="info-link">
            🔗 Learn More
          </a>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    book: {
      type: Object,
      required: true
    }
  },

  emits: ['close'],

  computed: {
    formattedAuthors() {
      if (!this.book || !this.book.authors) return ''
      const authors = typeof this.book.authors === 'string'
        ? this.book.authors.split(',').map(a => a.trim())
        : Array.isArray(this.book.authors) ? this.book.authors : []
      return authors.sort().join(', ')
    }
  },

  methods: {
    close() {
      this.$emit('close')
    }
  },

  mounted() {
    // Prevent body scroll when modal is open
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
  max-width: 600px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  position: relative;
  animation: slideUp 0.3s ease;
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
  z-index: 10;
}

.modal-close:hover {
  background-color: var(--bg-secondary);
  color: var(--text-primary);
}

.modal-header {
  display: flex;
  gap: 20px;
  padding: 30px;
  border-bottom: 2px solid var(--border-color);
}

.book-image {
  flex-shrink: 0;
}

.image-placeholder {
  width: 120px;
  height: 180px;
  background: linear-gradient(135deg, var(--primary-purple-light), var(--primary-pink-light));
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 48px;
  color: white;
}

.book-header-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}

.book-title {
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 12px 0;
  line-height: 1.3;
}

.book-authors {
  font-size: 14px;
  color: var(--text-secondary);
  margin: 0;
}

.modal-body {
  padding: 30px;
}

.detail-group {
  margin-bottom: 24px;
}

.detail-group:last-child {
  margin-bottom: 0;
}

.detail-label {
  display: block;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.detail-value {
  font-size: 14px;
  color: var(--text-primary);
  line-height: 1.6;
}

.detail-value.score {
  font-size: 18px;
  font-weight: 600;
  color: var(--primary-purple-dark);
}

.genres-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.genre-tag {
  display: inline-block;
  background-color: var(--bg-secondary);
  color: var(--text-primary);
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 13px;
  border: 1px solid var(--border-color);
}

.description {
  font-size: 14px;
  line-height: 1.7;
  color: var(--text-secondary);
}

.text-muted {
  color: var(--text-tertiary);
  font-style: italic;
}

.info-link {
  display: inline-block;
  color: var(--primary-purple-dark);
  text-decoration: none;
  font-weight: 600;
  padding: 8px 12px;
  border-radius: 6px;
  background-color: var(--bg-secondary);
  transition: all 0.2s ease;
}

.info-link:hover {
  background-color: var(--primary-purple-light);
  color: white;
}
</style>
