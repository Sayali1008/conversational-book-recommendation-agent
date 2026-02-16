<template>
    <div class="add-book-container">
        <div class="add-book-form">
            <h2 class="form-title">📚 Add New Book</h2>

            <form @submit.prevent="handleSubmit">
                <!-- Title -->
                <div class="form-group">
                    <label for="title" class="form-label">Title *</label>
                    <input id="title" v-model="form.title" type="text" class="form-input" placeholder="Enter book title"
                        required :disabled="loading" />
                </div>

                <!-- Description -->
                <div class="form-group">
                    <label for="description" class="form-label">Description *</label>
                    <textarea id="description" v-model="form.description" class="form-textarea"
                        placeholder="Enter book description (minimum 10 characters)" rows="4" required
                        :disabled="loading"></textarea>
                    <p class="form-help">{{ form.description.length }} / 10 characters minimum</p>
                </div>

                <!-- Authors -->
                <div class="form-group">
                    <label class="form-label">Authors *</label>
                    <div class="multi-select-wrapper">
                        <div class="multi-select-container">
                            <input v-model="authorSearch" type="text" class="form-input"
                                placeholder="Search or type author name..." @input="filterAuthors"
                                :disabled="loading" />
                            <div v-if="authorSearch && filteredAuthors.length" class="dropdown">
                                <div v-for="author in filteredAuthors" :key="author.author_id" class="dropdown-item"
                                    @click="selectAuthor(author)">
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
                            <input v-model="newAuthorName" type="text" class="form-input" placeholder="Author name"
                                @keyup.enter="confirmCreateAuthor" @keyup.esc="cancelCreateAuthor" autofocus />
                            <button type="button" class="btn-mini btn-success" @click="confirmCreateAuthor">✓</button>
                            <button type="button" class="btn-mini btn-cancel" @click="cancelCreateAuthor">✕</button>
                        </div>
                    </div>
                    <div v-if="form.authors.length" class="selected-items">
                        <div v-for="authorId in form.authors" :key="authorId" class="selected-item">
                            {{ getAuthorName(authorId) }}
                            <button type="button" class="remove-btn" @click="removeAuthor(authorId)"
                                :disabled="loading">
                                ✕
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Genres -->
                <div class="form-group">
                    <label class="form-label">Genres *</label>
                    <div class="multi-select-wrapper">
                        <div class="multi-select-container">
                            <input v-model="genreSearch" type="text" class="form-input"
                                placeholder="Search or type genre..." @input="filterGenres" :disabled="loading" />
                            <div v-if="genreSearch && filteredGenres.length" class="dropdown">
                                <div v-for="genre in filteredGenres" :key="genre.genre_id" class="dropdown-item"
                                    @click="selectGenre(genre)">
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
                            <input v-model="newGenreName" type="text" class="form-input" placeholder="Genre name"
                                @keyup.enter="confirmCreateGenre" @keyup.esc="cancelCreateGenre" autofocus />
                            <button type="button" class="btn-mini btn-success" @click="confirmCreateGenre">✓</button>
                            <button type="button" class="btn-mini btn-cancel" @click="cancelCreateGenre">✕</button>
                        </div>
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
                    <label for="infolink" class="form-label">Info Link</label>
                    <input id="infolink" v-model="form.infolink" type="url" class="form-input" placeholder="https://..."
                        :disabled="loading" />
                    <p class="form-help">(optional) Link to book details</p>
                </div>

                <!-- Submit -->
                <button type="submit" class="btn btn-primary" :disabled="loading || !isFormValid">
                    {{ loading ? '⏳ Adding...' : '➕ Add Book' }}
                </button>
            </form>
        </div>
    </div>
</template>

<script>
import { getAuthors, getGenres, createAuthor, createGenre, addBook } from '../services/api.js'

export default {
    data() {
        return {
            form: {
                title: '',
                authors: [],
                description: '',
                genres: [],
                infolink: ''
            },
            authorSearch: '',
            genreSearch: '',
            allAuthors: [],
            allGenres: [],
            filteredAuthors: [],
            filteredGenres: [],
            creatingAuthor: false,
            newAuthorName: '',
            creatingGenre: false,
            newGenreName: '',
            loading: false
        }
    },

    computed: {
        isFormValid() {
            return (
                this.form.title.trim().length > 0 &&
                this.form.authors.length > 0 &&
                this.form.description.length >= 10 &&
                this.form.genres.length > 0
            )
        }
    },

    methods: {
        async handleSubmit() {
            if (this.loading || !this.isFormValid) return
            this.loading = true

            try {
                const payload = {
                    title: this.form.title,
                    authors: this.form.authors,
                    description: this.form.description,
                    genres: this.form.genres,
                    infolink: this.form.infolink || null
                }

                const response = await addBook(payload)
                alert('Book added successfully!')
                this.resetForm()
                this.$emit('book-added', response)
            } catch (error) {
                console.error('Error adding book:', error)
                alert('Failed to add book: ' + error.message)
            } finally {
                this.loading = false
            }
        },

        resetForm() {
            this.form = {
                title: '',
                authors: [],
                description: '',
                genres: [],
                infolink: ''
            }
            this.authorSearch = ''
            this.genreSearch = ''
        },

        filterAuthors() {
            if (!this.authorSearch) {
                this.filteredAuthors = []
                return
            }
            this.filteredAuthors = this.allAuthors.filter(a =>
                a.name.toLowerCase().includes(this.authorSearch.toLowerCase()) &&
                !this.form.authors.includes(a.author_id)
            )
        },

        filterGenres() {
            if (!this.genreSearch) {
                this.filteredGenres = []
                return
            }
            this.filteredGenres = this.allGenres.filter(g =>
                g.name.toLowerCase().includes(this.genreSearch.toLowerCase()) &&
                !this.form.genres.includes(g.genre_id)
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

        startCreateAuthor() {
            this.newAuthorName = this.authorSearch
            this.creatingAuthor = true
        },

        async confirmCreateAuthor() {
            if (!this.newAuthorName.trim()) return

            try {
                const response = await createAuthor(this.newAuthorName)
                this.allAuthors.push(response)
                this.selectAuthor(response)
                this.creatingAuthor = false
                this.newAuthorName = ''
                this.authorSearch = ''
            } catch (error) {
                alert('Failed to create author: ' + error.message)
            }
        },

        cancelCreateAuthor() {
            this.creatingAuthor = false
            this.newAuthorName = ''
        },

        startCreateGenre() {
            this.newGenreName = this.genreSearch
            this.creatingGenre = true
        },

        async confirmCreateGenre() {
            if (!this.newGenreName.trim()) return

            try {
                const response = await createGenre(this.newGenreName)
                this.allGenres.push(response)
                this.selectGenre(response)
                this.creatingGenre = false
                this.newGenreName = ''
                this.genreSearch = ''
            } catch (error) {
                alert('Failed to create genre: ' + error.message)
            }
        },

        cancelCreateGenre() {
            this.creatingGenre = false
            this.newGenreName = ''
        },

        getAuthorName(authorId) {
            const author = this.allAuthors.find(a => a.author_id === authorId)
            return author ? author.name : 'Unknown'
        },

        getGenreName(genreId) {
            const genre = this.allGenres.find(g => g.genre_id === genreId)
            return genre ? genre.name : 'Unknown'
        }
    },

    async mounted() {
        try {
            const authorsRes = await getAuthors()
            this.allAuthors = authorsRes.authors

            const genresRes = await getGenres()
            this.allGenres = genresRes.genres
        } catch (error) {
            console.error('Error loading authors/genres:', error)
            alert('Failed to load authors and genres')
        }
    }
}
</script>

<style scoped>
.add-book-container {
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.add-book-form {
    background: white;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    padding: 3rem;
    max-width: 600px;
    width: 100%;
    max-height: 90vh;
    overflow-y: auto;
}

.form-title {
    font-size: 2rem;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 2rem;
    text-align: center;
}

.form-group {
    margin-bottom: 1.5rem;
}

.form-label {
    display: block;
    font-size: 0.95rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.form-input,
.form-textarea {
    width: 100%;
    padding: 0.75rem;
    border: 2px solid #e1e8ed;
    border-radius: 8px;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    font-family: inherit;
    box-sizing: border-box;
}

.form-input:focus,
.form-textarea:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.form-input:disabled,
.form-textarea:disabled {
    background-color: #f5f7fa;
    cursor: not-allowed;
}

.form-textarea {
    resize: vertical;
    min-height: 100px;
}

.form-help {
    font-size: 0.85rem;
    color: #7f8c8d;
    margin-top: 0.5rem;
}

.multi-select-wrapper {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.multi-select-container {
    position: relative;
}

.dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: white;
    border: 2px solid #667eea;
    border-top: none;
    border-radius: 0 0 8px 8px;
    max-height: 200px;
    overflow-y: auto;
    z-index: 10;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.dropdown-item {
    padding: 0.75rem;
    cursor: pointer;
    transition: background-color 0.2s;
    border-bottom: 1px solid #ecf0f1;
}

.dropdown-item:hover {
    background-color: #f0f4ff;
}

.dropdown-item:last-child {
    border-bottom: none;
}

.dropdown-item.add-new {
    color: #667eea;
    font-weight: 500;
    background-color: #f8faff;
}

.dropdown-item.add-new:hover {
    background-color: #f0f4ff;
}

.inline-create {
    display: flex;
    gap: 0.5rem;
    align-items: center;
}

.inline-create .form-input {
    flex: 1;
    margin: 0;
}

.btn-mini {
    padding: 0.5rem 0.75rem;
    border: none;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}

.btn-success {
    background-color: #27ae60;
    color: white;
}

.btn-success:hover {
    background-color: #229954;
}

.btn-cancel {
    background-color: #e74c3c;
    color: white;
}

.btn-cancel:hover {
    background-color: #c0392b;
}

.selected-items {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

.selected-item {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background-color: #f0f4ff;
    color: #667eea;
    padding: 0.5rem 0.75rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 500;
}

.remove-btn {
    background: none;
    border: none;
    color: #667eea;
    cursor: pointer;
    font-size: 1rem;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color 0.2s;
}

.remove-btn:hover:not(:disabled) {
    color: #764ba2;
}

.remove-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.btn {
    width: 100%;
    padding: 0.875rem;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
}

.btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}
</style>