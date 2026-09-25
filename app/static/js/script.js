/**
 * MovieLens ABSA - Frontend Common Script
 */

// Mobile Nav Toggle
function toggleMobileNav() {
  const nav = document.getElementById('navLinks');
  if(nav) nav.classList.toggle('open');
}

// Create Movie Card HTML (used across home and movies page)
function createMovieCard(movie) {
  const card = document.createElement('a');
  card.className = 'movie-card';
  card.href = `/movie/${movie.id}`;
  card.style.textDecoration = 'none';
  card.style.color = 'inherit';
  
  const posterUrl = movie.poster_url || '/static/img/placeholder.png';
  const year = movie.year ? `<div class="mc-year">${movie.year}</div>` : '';
  const genres = movie.genres && movie.genres.length > 0 ? `<div class="mc-genres">${movie.genres.slice(0,2).join(' • ')}</div>` : '';
  
  let edgeClass = 'rA';
  if(movie.edge_rating === 'A+') edgeClass = 'rA+';
  if(movie.edge_rating === 'B+') edgeClass = 'rB+';
  if(movie.edge_rating === 'B') edgeClass = 'rB';
  if(movie.edge_rating === 'C') edgeClass = 'rC';

  card.innerHTML = `
    <div class="mc-poster-wrap">
      <img src="${posterUrl}" alt="${movie.title}" class="mc-poster" onerror="this.src='/static/img/placeholder.png'; this.style.objectFit='contain';" loading="lazy" />
      <div class="mc-hover-overlay">
        <button class="mc-hover-btn">View Details</button>
      </div>
    </div>
    <div class="mc-content">
      <div class="mc-title" title="${movie.title}">${movie.title}</div>
      ${year}
      ${genres}
      <div class="mc-footer">
        <span class="rating-badge ${edgeClass}">${movie.edge_rating || 'N/A'}</span>
        <span class="mc-age">${movie.age_category || ''}</span>
      </div>
    </div>
  `;
  return card;
}

// ════════════ ABSA ANALYSIS ════════════
function updateCharCount() {
  const ta = document.getElementById('reviewInput');
  if(!ta) return;
  const cnt = ta.value.length;
  const cc = document.getElementById('charCount');
  if(cc) {
      cc.textContent = `${cnt} / 5000 characters`;
      cc.style.color = cnt > 5000 ? 'var(--danger)' : 'var(--text-muted)';
  }
}

const exampleReviews = [
  "The acting was phenomenal, especially by the lead actor. However, the plot felt incredibly slow and boring.",
  "Absolutely stunning visuals and cinematography! The music was breathtaking too. Highly recommended.",
  "The director did a terrible job, the pacing was all over the place. But I loved the special effects."
];

function loadExample(idx) {
  const ta = document.getElementById('reviewInput');
  if(ta) {
      ta.value = exampleReviews[idx];
      updateCharCount();
  }
}

async function analyzeReview() {
  const textInput = document.getElementById('reviewInput');
  if(!textInput) return;
  const text = textInput.value.trim();
  
  if(!text) {
    alert("Please enter a review to analyze.");
    return;
  }
  if(text.length > 5000) {
    alert("Review is too long. Please keep it under 5000 characters.");
    return;
  }
  
  document.getElementById('analyzeBtnText').style.display = 'none';
  document.getElementById('analyzeBtnSpinner').style.display = 'inline-block';
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('analyzeStatus').style.display = 'block';
  document.getElementById('resultsPanel').style.display = 'none';
  document.getElementById('errorPanel').style.display = 'none';
  document.getElementById('noAspectsCard').style.display = 'none';
  
  try {
    const res = await fetch(`/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_text: text })
    });
    
    if(!res.ok) throw new Error(`Server returned ${res.status}`);
    
    const data = await res.json();
    displayResults(data);
    
  } catch (err) {
    console.error("Analysis failed:", err);
    document.getElementById('errorPanel').style.display = 'block';
    document.getElementById('errorMsg').textContent = err.message;
  } finally {
    document.getElementById('analyzeBtnText').style.display = 'inline';
    document.getElementById('analyzeBtnSpinner').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('analyzeStatus').style.display = 'none';
  }
}

function displayResults(data) {
  document.getElementById('resultsPanel').style.display = 'block';
  
  const osBadge = document.getElementById('overallBadge');
  const overall = data.overall_sentiment;
  if(overall.sentiment === 'positive') {
    osBadge.textContent = '🟢 Positive';
    osBadge.style.color = 'var(--success)';
  } else {
    osBadge.textContent = '🔴 Negative';
    osBadge.style.color = 'var(--danger)';
  }
  
  const aspects = data.aspects || [];
  document.getElementById('totalAspects').textContent = aspects.length;
  
  if (aspects.length === 0) {
    document.getElementById('noAspectsCard').style.display = 'block';
    document.getElementById('aspectCardsContainer').style.display = 'none';
    document.getElementById('positiveCount').textContent = "0";
    document.getElementById('negativeCount').textContent = "0";
    return;
  }
  
  document.getElementById('noAspectsCard').style.display = 'none';
  const container = document.getElementById('aspectCardsContainer');
  container.style.display = 'grid';
  container.innerHTML = '<h3 class="mb-3" style="grid-column: 1/-1;">Detected Aspects</h3>';
  
  let pos = 0; let neg = 0;
  
  aspects.forEach(asp => {
    if(asp.sentiment === 'positive') pos++;
    else neg++;
    
    const card = document.createElement('div');
    card.className = 'aspect-result-card';
    
    const confPct = (asp.confidence * 100).toFixed(1);
    const badgeCls = asp.sentiment === 'positive' ? 'pos' : 'neg';
    const sentText = asp.sentiment.toUpperCase();
    
    card.innerHTML = `
      <div class="arc-header">
        <div>
          <div class="arc-aspect">${asp.aspect}</div>
          <div class="arc-cat">${asp.category}</div>
        </div>
        <div class="arc-badge ${badgeCls}">${sentText}</div>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; margin-top:5px; color:#94a3b8">
        <span>Confidence</span>
        <span>${confPct}%</span>
      </div>
      <div class="arc-conf-bar">
        <div class="arc-conf-fill" style="width: ${confPct}%; background: var(--${asp.sentiment === 'positive' ? 'success' : 'danger'})"></div>
      </div>
      <div class="arc-context">"${asp.context_sentence}"</div>
    `;
    container.appendChild(card);
  });
  
  document.getElementById('positiveCount').textContent = pos;
  document.getElementById('negativeCount').textContent = neg;
}
