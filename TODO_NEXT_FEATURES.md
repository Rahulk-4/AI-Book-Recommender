# TODO_NEXT_FEATURES

## Phase 1: Login + Favorites + Mood + Details Popup (approved)

### Step 1: Add session-based auth gate
- [ ] Implement login/signup UI in `app.py`
- [ ] Store users in `users.json` (hash passwords if possible; otherwise simple for demo)
- [ ] Add logout button

### Step 2: Favorites (wishlist)
- [ ] Add “❤️ Add to Favorites” button per book in Recommend + Search
- [ ] Persist favorites to `favorites.json` per username
- [ ] Add a Favorites view in Dashboard tab

### Step 3: Mood-based recommendation (unique feature)
- [ ] Add Mood selector in Recommend tab
- [ ] Implement mood->genre mapping + scoring/filtering
- [ ] Show 5 recommendations and allow “Add to Favorites”

### Step 4: Book details popup (professional)
- [ ] Add “Details” button per recommended book
- [ ] Display title/author/genre/rating/description/year(if available)/image
- [ ] Keep existing attractive UI

### Step 5: Responsive polish
- [ ] Add basic responsive CSS for card grid

### Step 6: Test locally
- [ ] Run `streamlit run app.py`
- [ ] Verify auth, favorites persistence, mood recs, details popup


