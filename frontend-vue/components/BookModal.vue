<template>
  <div class="modal-overlay" @click="$emit('close')">
    <div class="modal-card" @click.stop>
      <button class="modal-close" @click="$emit('close')">×</button>

      <div v-if="book" class="modal-content">
        <h2 class="modal-title">{{ book.title }}</h2>

        <p class="modal-id" v-if="book.book_id">Book ID: {{ book.book_id }}</p>

        <div v-if="book.authors && book.authors.length" class="modal-authors">
          <span class="modal-label">By:</span>
          {{ book.authors.join(', ') }}
        </div>

        <div v-if="book.genres && book.genres.length" class="modal-genres">
          <span class="genre-tag" v-for="genre in book.genres" :key="genre">
            {{ genre }}
          </span>
        </div>

        <div class="modal-description">
          <h3 class="modal-section-title">Description</h3>
          <p>{{ book.description || 'No description available.' }}</p>
        </div>

        <div v-if="book.infolink" class="modal-link">
          <a
            :href="book.infolink"
            target="_blank"
            rel="noopener noreferrer"
            class="btn btn-link"
          >
            <span class="btn-icon">🔗</span>
            View on Google Books
          </a>
        </div>

        <div class="modal-actions">
          <button @click="$emit('dislike')" class="btn btn-dislike" :disabled="loading">
            <span class="btn-icon-large">👎</span>
            Not Interested
          </button>
          <button @click="$emit('like')" class="btn btn-like" :disabled="loading">
            <span class="btn-icon-large">👍</span>
            I Like This!
          </button>
        </div>
      </div>

      <div v-else class="modal-loading">
        <div class="spinner-large"></div>
        <p>Loading book details...</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    book: { type: Object, default: null },
    loading: { type: Boolean, default: false }
  },
  emits: ['close', 'like', 'dislike']
}
</script>
