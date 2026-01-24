<template>
  <div v-if="!books.length" class="empty-state">
    <div class="empty-icon">📖</div>
    <h3>No recommendations yet</h3>
    <p>Get started by clicking "Get Recommendations" above</p>
  </div>

  <div v-else class="recommendations-grid">
    <div
      v-for="(book, idx) in books"
      :key="book.book_id"
      class="book-card"
      :style="{ animationDelay: `${idx * 50}ms` }"
    >
      <!-- Book Cover -->
      <div class="book-cover">
        <button
          class="cover-button"
          @click="$emit('show-modal', book)"
          :disabled="disabled"
        >
          <div class="cover-placeholder">
            <span class="cover-icon">📕</span>
          </div>
        </button>
      </div>

      <!-- Book Info -->
      <div class="book-info">
        <button
          class="book-title-button"
          @click="$emit('show-modal', book)"
          :disabled="disabled"
        >
          {{ book.title }}
        </button>
        <div v-if="book.authors && book.authors.length" class="book-authors">
          {{ book.authors.join(', ') }}
        </div>

        <div class="book-meta">
          <span class="meta-badge score">⭐ {{ book.score.toFixed(2) }}</span>
          <span class="meta-badge source">{{ book.source }}</span>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="book-actions">
        <button
          class="action-btn dislike"
          @click="$emit('dislike', book)"
          :disabled="disabled"
          title="I don't like this"
        >
          <span>👎</span>
        </button>
        <button
          class="action-btn like"
          @click="$emit('like', book)"
          :disabled="disabled"
          title="I like this!"
        >
          <span>👍</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    books: { type: Array, default: () => [] },
    disabled: { type: Boolean, default: false }
  },
  emits: ['show-modal', 'like', 'dislike']
}
</script>
