const API_BASE = 'http://localhost:8000'

export async function checkRecommendationStatus() {
    const res = await fetch(`${API_BASE}/recommendation/status`)
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
