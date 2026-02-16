const API_BASE = 'http://localhost:8000'

// ==================== Users ====================

export async function createUser(payload) {
  const res = await fetch(`${API_BASE}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!res.ok) {
    const errorData = await res.json()
    throw new Error(errorData.detail || 'Registration failed')
  }
  return res.json()
}

export async function loginUser(userId) {
  const res = await fetch(`${API_BASE}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId })
  })
  if (!res.ok) {
    const errorData = await res.json()
    throw new Error(errorData.detail || 'Login failed')
  }
  return res.json()
}

export async function checkUserIdExists(userId) {
  const res = await fetch(`${API_BASE}/users/${userId}`)
  if (res.status === 404) return false
  if (!res.ok) throw new Error('Failed to check user ID')
  return true
}

// ==================== Genres ====================

export async function getGenres() {
  const res = await fetch(`${API_BASE}/genres`)
  if (!res.ok) throw new Error('Failed to fetch genres')
  return res.json()
}

export async function createGenre(name) {
  const res = await fetch(`${API_BASE}/genres`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  })
  if (!res.ok) {
    const errorData = await res.json()
    throw new Error(errorData.detail || 'Failed to create genre')
  }
  return res.json()
}

export async function saveUserGenres(payload) {
  const res = await fetch(`${API_BASE}/preferred-genres`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!res.ok) {
    const errorData = await res.json()
    throw new Error(errorData.detail || 'Failed to save genres')
  }
  return res.json()
}

// ==================== Authors ====================

export async function getAuthors(q) {
  const qs = q ? `?q=${encodeURIComponent(q)}` : ''
  const res = await fetch(`${API_BASE}/authors${qs}`)
  if (!res.ok) throw new Error('Failed to fetch authors')
  return res.json()
}

export async function createAuthor(name) {
  const res = await fetch(`${API_BASE}/authors`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  })
  if (!res.ok) {
    const errorData = await res.json()
    throw new Error(errorData.detail || 'Failed to create author')
  }
  return res.json()
}

// ==================== Books ====================

export async function addBook(payload) {
  const res = await fetch(`${API_BASE}/books`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!res.ok) {
    const errorData = await res.json()
    let errorMessage = 'Failed to add book'
    
    if (errorData.detail) {
      // Handle validation errors (array)
      if (Array.isArray(errorData.detail)) {
        const messages = errorData.detail.map(err => {
          const field = err.loc ? err.loc[err.loc.length - 1] : 'form'
          return `${field}: ${err.msg}`
        })
        errorMessage = messages.join(' | ')
      } else {
        // Handle simple error message
        errorMessage = errorData.detail
      }
    }
    throw new Error(errorMessage)
  }
  return res.json()
}

// ==================== Recommendations ====================

export async function checkRecommendationStatus() {
  const res = await fetch(`${API_BASE}/status`)
  if (!res.ok) throw new Error('Status check failed')
  return res.json()
}

export async function runPipeline() {
  const res = await fetch(`${API_BASE}/pipeline/run`, { method: 'POST' })
  if (!res.ok) throw new Error('Pipeline start failed')
  return res.json()
}

export async function fetchRecommendations(params) {
  const qs = new URLSearchParams(params).toString()
  const res = await fetch(`${API_BASE}/recommend?${qs}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function fetchBook(bookId) {
  const res = await fetch(`${API_BASE}/book/${bookId}`)
  if (!res.ok) throw new Error('Book fetch failed')
  return res.json()
}

export async function searchBooks(title, authorIds) {
  const params = new URLSearchParams()
  params.append('title', title)
  if (authorIds && authorIds.length > 0) {
    authorIds.forEach(id => params.append('author_ids', id))
  }
  const url = `${API_BASE}/books/search?${params.toString()}`
  console.log('Searching books - URL:', url)
  const res = await fetch(url)
  if (!res.ok) return { books: [] }
  return res.json()
}

export async function fetchPipelineStatus() {
  const res = await fetch(`${API_BASE}/pipeline/status`)
  if (!res.ok) throw new Error('Pipeline status fetch failed')
  return res.json()
}

export async function swipe(payload) {
  const res = await fetch(`${API_BASE}/swipe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
