# Catalonia Wildfire Detection System

## Risks & Mitigations

**Project Type:** Postgraduate Capstone Project
**Domain:** Computer Vision / Remote Sensing
**Region:** Catalonia, Spain
**Date:** January 2026

---

Risk management identifies potential problems that could derail the project and develops strategies to prevent or mitigate them. For a capstone project, proactive risk management ensures timely completion and quality deliverables.

**Risk Management Philosophy:**

- **Proactive identification:** Identify risks early, before they become problems
- **Prioritization:** Focus on high-impact, high-probability risks
- **Practical mitigation:** Choose mitigation strategies that are feasible for capstone scope
- **Continuous monitoring:** Review risks regularly and adjust strategies as needed

---

### 1. Technical Risks

Technical risks are problems related to technology, tools, or implementation that could prevent the system from working correctly or meeting performance targets.

#### 1.1 Model Performance Risks

**Risk: Model accuracy below target (IoU < 0.70)**

**Impact:** High - Model performance is the primary success criterion for capstone

**Probability:** Medium - Deep learning performance can be unpredictable

**Why this risk exists:**
- Insufficient training data or poor data quality
- Model architecture not suitable for task
- Hyperparameters not optimized
- Class imbalance not handled properly
- Overfitting to training data

**Mitigation Strategies:**

**Prevention:**
- **Use proven architecture:** U-Net with pretrained encoder (Section 3.1) - well-established for segmentation
- **Comprehensive data exploration:** Identify and fix data quality issues early (Section 2.8)
- **Proper class imbalance handling:** Implement loss weighting, weighted sampling, or focal loss (Section 2.6)
- **Geographic validation split:** Ensure model generalizes, not just memorizes training regions (Section 2.5)
- **Catalonia validation set:** Test transfer learning early (Section 8.2) - catch geographic bias before final evaluation

**Response (if risk occurs):**
- **Iterate on data augmentation:** Increase augmentation diversity, adjust parameters
- **Hyperparameter tuning:** Run more Optuna trials, expand search space
- **Try different encoder:** Switch from ResNet-34 to EfficientNet-B0 or vice versa
- **Ensemble models:** Combine multiple models if single model insufficient
- **Fine-tune on Catalonia data:** If transfer learning fails, fine-tune on Catalonia validation set
- **Adjust success criteria:** If IoU consistently 0.65-0.70, document why and focus on other metrics

**Early warning signs:**
- Validation IoU plateaus below 0.65 after 20+ epochs
- Large gap between training and validation metrics (overfitting)
- Poor performance on Catalonia validation set (geographic bias)

---

**Risk: Model training instability (loss doesn't converge, NaN values)**

**Impact:** High - Training failure prevents model development

**Probability:** Low - Modern frameworks handle this well, but can occur

**Why this risk exists:**
- Learning rate too high
- Gradient explosion
- Numerical instability in loss function
- Data preprocessing errors (NaN/Inf values)

**Mitigation Strategies:**

**Prevention:**
- **Learning rate schedule:** Use ReduceLROnPlateau to automatically reduce LR (Section 3.2)
- **Gradient clipping:** Implement gradient clipping (max_norm=1.0) to prevent explosion
- **Data validation:** Check for NaN/Inf in preprocessing pipeline (Section 2.8)
- **Loss function stability:** Use combined BCE+Dice loss (more stable than Dice alone)
- **Mixed precision training:** Use automatic mixed precision (AMP) for numerical stability

**Response (if risk occurs):**
- **Reduce learning rate:** Lower initial LR (1e-5 instead of 1e-4)
- **Check data:** Verify no NaN/Inf in input data
- **Simplify model:** Temporarily use smaller model to test stability
- **Check loss function:** Verify loss calculation is correct

---

#### 1.2 Infrastructure & Integration Risks

**Risk: Sentinel API rate limits or access issues**

**Impact:** Medium - Prevents live data fetching, but can work with cached data

**Probability:** Medium - Free tier APIs often have rate limits

**Why this risk exists:**
- Copernicus Data Space API has rate limits for free accounts
- Sentinel Hub API requires account setup and may have limits
- Network issues or API downtime

**Mitigation Strategies:**

**Prevention:**
- **Implement caching:** Cache fetched imagery to avoid repeated API calls
- **Request queuing:** Queue requests and add delays between calls
- **Multiple API options:** Support both Copernicus Data Space and Sentinel Hub (fallback)
- **Test API access early:** Verify API access in Phase 1, not Phase 3
- **Use batch requests:** Request multiple dates/regions in single call when possible

**Response (if risk occurs):**
- **Use cached data:** For demo, use pre-downloaded imagery instead of live fetch
- **Reduce API calls:** Cache more aggressively, reuse imagery for multiple tests
- **Alternative data source:** Use downloaded Sentinel-2 imagery from training datasets
- **Simplify demo:** Show detection on pre-loaded imagery instead of live fetch

---

**Risk: Integration complexity (APIs, libraries don't work together)**

**Impact:** Medium - Delays development, but workarounds exist

**Probability:** Medium - Multiple libraries and APIs increase integration risk

**Why this risk exists:**
- Multiple Python libraries with version conflicts
- API authentication and setup complexity
- Coordinate system conversions between libraries
- Different data formats (raster, vector, GeoJSON)

**Mitigation Strategies:**

**Prevention:**
- **Use well-documented APIs:** Copernicus Data Space and Sentinel Hub have good documentation
- **Build incrementally:** Test each component separately before integrating
- **Version pinning:** Pin library versions in requirements.txt to avoid conflicts
- **Isolated testing:** Test API integration in separate script before adding to main code
- **Standard formats:** Use standard formats (GeoJSON, WGS84) to minimize conversions

**Response (if risk occurs):**
- **Simplify integration:** Use simpler library or manual implementation
- **Mock external services:** Use mock data for development, integrate later
- **Seek help:** Use library documentation, Stack Overflow, or community forums
- **Alternative libraries:** Switch to alternative library if primary doesn't work

---

**Risk: Deployment platform issues (GCP Cloud Run)**

**Impact:** Medium - Can use alternative deployment, but adds time

**Probability:** Low - GCP Cloud Run is stable, but can have issues

**Why this risk exists:**
- Platform downtime or maintenance
- Resource limits (CPU, memory, storage)
- Build failures due to dependencies or configuration
- Credit limits or billing issues

**Mitigation Strategies:**

**Prevention:**
- **Test deployment early:** Deploy minimal version early to test platform
- **Document dependencies:** Clear requirements.txt, test locally first
- **Resource monitoring:** Check resource usage, optimize if needed
- **Credit monitoring:** Set up budget alerts to track credit usage
- **Alternative ready:** Have backup deployment option (Streamlit Cloud, local Docker)

**Response (if risk occurs):**
- **Use alternative platform:** Switch to Streamlit Cloud or local Docker
- **Simplify deployment:** Reduce dependencies, use smaller model
- **Local demo:** Deploy locally for demo if cloud deployment fails
- **Check credits:** Verify credit availability, contact university if needed

---

#### 1.3 Resource & Cost Risks

**Risk: GCP credit usage exceeds available credits**

**Impact:** Medium - May need to switch to free alternatives or request more credits

**Probability:** Low - Credits should be sufficient, but usage can be unpredictable

**Why this risk exists:**
- Accidental high resource usage
- Misconfiguration leading to excessive costs
- Extended development/testing period
- Multiple deployments or rebuilds

**Mitigation Strategies:**

**Prevention:**
- **Set budget alerts:** Configure GCP budget alerts to monitor credit usage
- **Monitor usage:** Check resource usage regularly in GCP console
- **Optimize resources:** Use appropriate instance sizes, scale to zero when not in use
- **Clean up resources:** Delete unused deployments, old images, test resources
- **Estimate costs:** Calculate expected costs before deployment

**Response (if risk occurs):**
- **Request more credits:** Contact university for additional credits if needed
- **Switch to free alternatives:** Use Streamlit Cloud or local Docker if credits exhausted
- **Reduce usage:** Optimize deployment, use smaller instances, reduce rebuilds
- **Local resources:** Use local GPU or CPU instead of cloud

---

**Risk: GPU availability for training**

**Impact:** High - Training without GPU is very slow

**Probability:** Low - Multiple free GPU options available

**Why this risk exists:**
- Google Colab GPU quotas may be limited
- University GPU resources may be unavailable
- Local GPU may not be available

**Mitigation Strategies:**

**Prevention:**
- **Multiple GPU options:** Identify Google Colab, university resources, local GPU
- **Start training early:** Begin training as soon as data is ready
- **Optimize training:** Use smaller batch sizes, fewer epochs if GPU limited
- **CPU fallback:** Can train on CPU (slow but possible)

**Response (if risk occurs):**
- **Use Google Colab:** Free GPU available (may have quotas)
- **Reduce training:** Smaller model, fewer hyperparameter trials
- **CPU training:** Accept slower training, start earlier
- **University resources:** Request access to university GPU cluster

---

### 2. Data Risks

Data risks are problems related to datasets, data quality, or data availability that could prevent successful model training or evaluation.

#### 2.1 Data Quality & Generalization Risks

**Risk: Training data doesn't generalize to Catalonia**

**Impact:** High - Model must work on Catalonia (project requirement)

**Probability:** Medium - Geographic transfer learning can fail

**Why this risk exists:**
- Training data from different regions (not Catalonia)
- Different vegetation, climate, fire characteristics
- Geographic bias in training data

**Mitigation Strategies:**

**Prevention:**
- **Catalonia validation set (mandatory):** Create and test early (Section 8.2)
- **Geographic validation split:** Ensure test set includes diverse regions (Section 2.5)
- **Include Spanish data:** Use CEMS-Wildfire which includes Spanish fires
- **Data augmentation:** Augment with geographic diversity (rotation, scaling)
- **Monitor geographic performance:** Track performance by region during training

**Response (if risk occurs):**
- **Fine-tune on Catalonia data:** Fine-tune model on Catalonia validation set
- **Increase Catalonia data:** Add more Catalonia-specific training data
- **Transfer learning adjustments:** Adjust learning rate, freeze/unfreeze layers
- **Document limitations:** If performance lower on Catalonia, document why and analyze

---

**Risk: Dataset quality issues (annotation errors, misalignment)**

**Impact:** Medium - Poor data quality reduces model performance

**Probability:** Low - Datasets are well-curated, but errors can exist

**Why this risk exists:**
- Human annotation errors
- Misalignment between imagery and masks
- Incorrect coordinate systems
- Missing or corrupted data

**Mitigation Strategies:**

**Prevention:**
- **Thorough data exploration:** Comprehensive quality checks (Section 2.8)
  - Visual inspection of random samples
  - Check mask alignment with imagery
  - Verify coordinate systems
  - Check for missing data
- **Use authoritative sources:** CEMS-Wildfire from Copernicus (official source)
- **Validate preprocessing:** Verify preprocessing doesn't introduce errors
- **Document data issues:** Record any quality issues found

**Response (if risk occurs):**
- **Clean problematic data:** Remove or fix erroneous samples
- **Use alternative dataset:** Switch to different dataset if primary has issues
- **Manual correction:** Manually correct critical errors if feasible
- **Document limitations:** Note data quality issues in final report

---

#### 2.2 Data Availability Risks

**Risk: Cloud cover limits available imagery**

**Impact:** Medium - Reduces available training/evaluation data

**Probability:** Medium - Cloud cover is common in satellite imagery

**Why this risk exists:**
- Sentinel-2 imagery often has cloud cover
- Catalonia validation set may have limited clear imagery
- Live detection may encounter cloudy dates

**Mitigation Strategies:**

**Prevention:**
- **Cloud filtering:** Filter out heavily clouded images (Section 2.8)
- **Multiple dates:** Use multiple dates per fire event (not just single date)
- **Cloud masks:** Use provided cloud masks to exclude clouded pixels
- **Temporal flexibility:** Allow date range selection, not just single date

**Response (if risk occurs):**
- **Use alternative dates:** Select different dates with less cloud cover
- **Accept partial cloud:** Train/evaluate on partially clouded images
- **Cloud-aware evaluation:** Evaluate only on clear pixels (using cloud masks)
- **Document cloud impact:** Analyze and document how cloud cover affects performance

---

**Risk: Dataset download/access issues**

**Impact:** Medium - Delays data preparation phase

**Probability:** Low - Datasets are publicly available, but access can be slow

**Why this risk exists:**
- Large dataset sizes (slow downloads)
- HuggingFace dataset viewer limitations
- Network issues or server downtime
- Access restrictions or authentication issues

**Mitigation Strategies:**

**Prevention:**
- **Start downloads early:** Begin downloading datasets in Week 1
- **Multiple sources:** Have backup datasets ready (CEMS-Wildfire, EO4WildFires)
- **Prioritize core dataset:** Focus on CEMS-Wildfire first (primary dataset)
- **Use university network:** May have faster download speeds
- **Download in background:** Download while working on other tasks

**Response (if risk occurs):**
- **Use smaller subset:** Work with available data, download more later
- **Alternative dataset:** Switch to alternative dataset if primary unavailable
- **Manual download:** Use browser or wget if HuggingFace API fails
- **Extend Phase 1:** Add time buffer if downloads are slow

---

### 3. Schedule Risks

Schedule risks are problems that could cause project delays, preventing timely completion of deliverables.

#### 3.1 Development Timeline Risks

**Risk: Model training takes longer than expected**

**Impact:** Medium - Delays model development, but buffers included

**Probability:** Medium - Training time is hard to predict

**Why this risk exists:**
- Hyperparameter tuning requires many trials
- Model may need more epochs to converge
- GPU availability may be limited
- Iterative improvements require multiple training runs

**Mitigation Strategies:**

**Prevention:**
- **Start training early:** Begin as soon as data is ready (Week 4)
- **Use GPU:** Significantly faster than CPU training
- **Limit search space:** Focused hyperparameter ranges (Section 3.4)
- **Early stopping:** Stop training if no improvement (prevents wasted time)
- **Buffer time:** Extra week included in Phase 2 (4 weeks instead of 3)

**Response (if risk occurs):**
- **Reduce hyperparameter trials:** 20 trials instead of 50 if time limited
- **Use smaller model:** Faster training, may still meet targets
- **Parallel training:** Train multiple models simultaneously if resources allow
- **Accept current performance:** If close to target, document and proceed

---

**Risk: Data preparation takes longer than expected**

**Impact:** Medium - Delays entire project timeline

**Probability:** Medium - Data preparation is complex and time-consuming

**Why this risk exists:**
- Multiple datasets to download and process
- Data exploration is comprehensive (10 checklist items)
- Catalonia validation set creation is detailed
- Preprocessing pipeline is complex

**Mitigation Strategies:**

**Prevention:**
- **Extended Phase 1:** 3 weeks allocated (instead of 2)
- **Start immediately:** Begin data exploration while downloading
- **Prioritize core tasks:** Focus on essential datasets and tasks first
- **Parallel work:** Data exploration can happen during downloads
- **Incremental approach:** Process one dataset at a time, don't wait for all

**Response (if risk occurs):**
- **Simplify data exploration:** Focus on critical checks, skip less important ones
- **Use smaller datasets:** Work with subset if full dataset too large
- **Defer non-critical tasks:** Postpone advanced preprocessing if time limited
- **Extend Phase 1:** Use buffer from later phases if needed

---

**Risk: Learning curve for new tools (W&B, Optuna, FastAPI, Streamlit)**

**Impact:** Medium - Delays development, but tools are relatively simple

**Probability:** High - Team may not be familiar with all tools

**Why this risk exists:**
- Multiple new tools to learn (W&B, Optuna, FastAPI, Streamlit, geospatial libraries)
- Limited time to learn each tool
- Tool-specific issues or bugs

**Mitigation Strategies:**

**Prevention:**
- **Choose simple tools:** W&B, Streamlit are user-friendly (Section 3.5, Section 6.1)
- **Start early:** Set up W&B in Phase 1, learn tools incrementally
- **Good documentation:** All recommended tools have excellent documentation
- **Tutorials:** Use official tutorials and examples
- **Allocate extra time:** Phase 1 has extra week for learning

**Response (if risk occurs):**
- **Simplify tool usage:** Use basic features only, skip advanced features
- **Alternative tools:** Switch to simpler alternatives if needed
- **Seek help:** Use documentation, tutorials, Stack Overflow
- **Pair programming:** Team members can help each other learn

---

#### 3.2 Scope & Feature Risks

**Risk: Scope creep (adding too many features)**

**Impact:** Medium - Delays completion, reduces focus on ML work

**Probability:** Medium - Temptation to add features is common

**Why this risk exists:**
- Desire to make system more impressive
- Adding features seems easy but takes time
- Unclear priorities between ML and application features

**Mitigation Strategies:**

**Prevention:**
- **Clear priorities:** ML work is primary focus (Section 1.1)
- **Simplified features:** API and UI are simplified for capstone (Sections 5, 6)
- **Feature list:** Stick to planned features, defer others
- **Regular reviews:** Review progress weekly, catch scope creep early

**Response (if risk occurs):**
- **Defer non-essential features:** Postpone advanced features to "future work"
- **Simplify existing features:** Reduce complexity of planned features
- **Focus on ML:** Prioritize model quality over UI polish
- **Document deferred features:** Note what was planned but not implemented

---

**Risk: Testing and documentation takes longer**

**Impact:** Medium - Delays final deliverables, but buffers included

**Probability:** High - Testing and documentation always take longer than expected

**Why this risk exists:**
- Comprehensive testing is time-consuming
- Documentation requires careful writing
- Bug fixes discovered during testing
- Evaluation report is extensive

**Mitigation Strategies:**

**Prevention:**
- **Extended Phase 6:** 2 weeks allocated for testing and documentation
- **Incremental documentation:** Write documentation during development, not just at end
- **Early testing:** Start testing components as they're completed
- **Template preparation:** Prepare report templates early

**Response (if risk occurs):**
- **Prioritize critical documentation:** Focus on evaluation report, model card
- **Simplify testing:** Focus on essential tests, skip exhaustive edge cases
- **Accept minor bugs:** Fix critical bugs only, document known issues
- **Use templates:** Use provided templates to speed up documentation

---

### 4. Academic & Quality Risks

Academic risks are problems related to meeting capstone requirements, academic standards, or evaluation criteria.

#### 4.1 Evaluation & Success Criteria Risks

**Risk: Model doesn't meet success criteria (IoU < 0.70)**

**Impact:** High - Primary success criterion not met

**Probability:** Medium - Target is achievable but not guaranteed

**Why this risk exists:**
- Success criteria are ambitious but realistic
- Model performance can be unpredictable
- Data or training issues may prevent reaching target

**Mitigation Strategies:**

**Prevention:**
- **Proven architecture:** U-Net is well-established for segmentation
- **Comprehensive training:** Proper hyperparameter tuning, data augmentation
- **Early validation:** Test on validation set regularly during training
- **Catalonia validation:** Test transfer learning early (catch issues before final evaluation)

**Response (if risk occurs):**
- **Document thoroughly:** Explain why target wasn't met, what was tried
- **Focus on other metrics:** Highlight strong performance on other metrics (Dice, Precision, Recall)
- **Error analysis:** Comprehensive error analysis shows understanding despite lower IoU
- **Baseline comparison:** Show improvement over baselines (demonstrates value)
- **Academic discussion:** Discuss limitations, future improvements (shows critical thinking)

---

**Risk: Insufficient documentation for reproducibility**

**Impact:** Medium - Reduces academic rigor, but can be fixed

**Probability:** Low - Documentation is planned, but may be incomplete

**Why this risk exists:**
- Documentation is time-consuming
- May forget to document some steps
- Reproducibility requires detailed documentation

**Mitigation Strategies:**

**Prevention:**
- **Reproducibility guide:** Planned in Phase 6 (Section 9.2)
- **Incremental documentation:** Document as you go, not just at end
- **Version control:** Use Git to track code changes
- **Data versioning:** Document data versions and preprocessing (Section 2.7)
- **Configuration files:** Save all hyperparameters and configs

**Response (if risk occurs):**
- **Add missing documentation:** Fill gaps in reproducibility guide
- **Code comments:** Add comments to code explaining key steps
- **README:** Comprehensive README with setup instructions
- **Document known issues:** Note any reproducibility limitations

---

### 5. Risk Monitoring & Response

**How to Monitor Risks:**

**Weekly risk review:**
- Review project progress against timeline
- Identify new risks as they emerge
- Check early warning signs (see main specification document, Section 9.5)
- Adjust mitigation strategies if needed

**Key indicators to watch:**
- **Week 3:** Data preparation progress (should be 80% complete)
- **Week 6:** Model training progress (should have initial trained model)
- **Week 10:** UI development progress (should have working prototype)
- **Week 13:** Testing progress (should have most tests complete)

**When to escalate:**
- **High-impact risks:** If high-impact risk occurs, prioritize mitigation immediately
- **Cascading risks:** If one risk causes others, address root cause
- **Timeline delays:** If more than 1 week behind, adjust scope or timeline

**Response procedures:**

**Immediate response:**
1. Identify the risk that occurred
2. Assess impact and urgency
3. Implement mitigation strategy (from risk table)
4. Monitor effectiveness
5. Adjust if needed

**Documentation:**
- Document any risks that occurred
- Note how they were mitigated
- Include in final report if relevant
- Learn from risks for future projects

---

### 6. Risk Summary Table

| Risk Category | Highest Priority Risks | Mitigation Status |
|---------------|----------------------|-------------------|
| **Technical** | Model accuracy below target | ✅ Multiple prevention strategies, early validation |
| **Technical** | API rate limits | ✅ Caching, multiple API options |
| **Technical** | GCP credit usage | ✅ Budget alerts, resource monitoring |
| **Data** | Training data doesn't generalize | ✅ Catalonia validation set (mandatory), early testing |
| **Data** | Cloud cover limits imagery | ✅ Cloud filtering, multiple dates |
| **Schedule** | Training takes longer | ✅ Buffer time, GPU usage, early start |
| **Schedule** | Learning curve for tools | ✅ Simple tools chosen, extra time allocated |
| **Academic** | Success criteria not met | ✅ Proven architecture, comprehensive training |

**Overall Risk Assessment:**

**Project risk level: Medium**
- Most risks have effective mitigation strategies
- Buffers included in timeline
- Multiple alternatives for critical components
- Focus on ML work (reduces application complexity risks)

**Confidence in success: High**
- Well-planned timeline with buffers
- Proven technologies and architectures
- Comprehensive risk mitigation strategies
- Clear priorities and scope
