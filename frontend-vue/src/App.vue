<!-- filepath: /Users/sayalimoghe/Documents/Career/GitHub/conversational-book-recommendation-agent/frontend-vue/src/App.vue -->
<template>
  <div
    style="max-width: 800px; margin: 24px auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;">
    <h2>Conversational Book Recommender</h2>

    <section style="margin-bottom: 16px;">
      <label>User ID:
        <input v-model="userId" placeholder="optional for cold-start" />
      </label>
      <label style="margin-left: 12px;">Seed Book IDs:
        <input v-model="seedBookIds" placeholder="e.g., 101,202,303" />
      </label>
      <label style="margin-left: 12px;">K:
        <input v-model.number="k" type="number" min="1" max="50" />
      </label>
      <button @click="fetchRecommendations" style="margin-left: 12px;">Get Recommendations</button>
    </section>

    <p v-if="strategy">Strategy: {{ strategy }}</p>

    <div v-if="recs.length" style="display: grid; grid-template-columns: 1fr; gap: 12px;">
      <div v-for="(r, idx) in recs" :key="r.book_id" style="border: 1px solid #ddd; padding: 12px; border-radius: 8px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <strong>{{ idx + 1 }}. {{ r.title }}</strong>
            <div v-if="r.authors && r.authors.length" style="font-size: 13px; color: #444; margin-top: 4px;">
              by {{ r.authors.join(', ') }}
            </div>
            <div style="font-size: 12px; color: #666;">Score: {{ r.score.toFixed(3) }} • Source: {{ r.source }}</div>
          </div>
          <div>
            <button @click="swipe(r, 'dislike')" style="margin-right: 8px;">👎</button>
            <button @click="swipe(r, 'like')">👍</button>
          </div>
        </div>
      </div>
    </div>

    <p v-else>No recommendations yet.</p>
  </div>
</template>

<script>
export default {
  data() {
    return {
      apiBase: 'http://localhost:8000',
      userId: '',
      seedBookIds: '',
      k: 10,
      recs: [],
      strategy: ''
    }
  },
  methods: {
    async fetchRecommendations() {
      const params = new URLSearchParams()
      if (this.userId) params.append('user_id', this.userId)
      params.append('k', String(this.k))
      if (this.seedBookIds) params.append('seed_book_ids', this.seedBookIds)
      const url = `${this.apiBase}/recommend?${params.toString()}`
      const res = await fetch(url)
      if (!res.ok) {
        const err = await res.text()
        throw new Error(err || 'Request failed')
      }
      const data = await res.json()
      this.recs = data.recommendations || []
      this.strategy = data.strategy || ''
    },
    async swipe(rec, action) {
      const payload = {
        user_id: this.userId || 'anonymous',
        book_id: rec.book_id,
        action,
        confidence: 1.0
      }
      const res = await fetch(`${this.apiBase}/swipe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (!res.ok) {
        const err = await res.text()
        throw new Error(err || 'Swipe failed')
      }
      const data = await res.json()
      if (data.next_recommendations) {
        this.recs = data.next_recommendations
        this.strategy = '' // next batch is generic; refresh if needed
      }
    }
  }
}
</script>