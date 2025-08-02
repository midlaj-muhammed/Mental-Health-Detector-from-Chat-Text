# Streamlit Cloud Deployment Guide

This guide helps you deploy the Mental Health Detector to Streamlit Cloud.

## 🚀 Quick Deployment Steps

### 1. Repository Setup
Ensure your GitHub repository contains these essential files:
- ✅ `streamlit_app.py` (main entry point for Streamlit Cloud)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `packages.txt` (system dependencies)
- ✅ All source code in `src/` directory

### 2. Streamlit Cloud Deployment

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Connect your GitHub account**
3. **Deploy new app** with these settings:
   - **Repository**: `midlaj-muhammed/Mental-Health-Detector-AI-Application-Built`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. **Click "Deploy!"**

### 3. Configuration Files

#### `streamlit_app.py` (Entry Point)
This file handles imports and error recovery for cloud deployment.

#### `requirements.txt` (Python Dependencies)
Contains all necessary Python packages with version constraints.

#### `packages.txt` (System Dependencies)
```
build-essential
curl
git
```

## 🔧 Troubleshooting Common Issues

### Issue 1: "No module named 'src.models'"

**Solution**: The app now includes fallback import strategies:
```python
try:
    from src.utils.analysis_engine import AnalysisEngine
except ImportError:
    from utils.analysis_engine import AnalysisEngine
```

### Issue 2: Model Download Failures

**Cause**: AI models need to be downloaded on first run
**Solution**: The app will automatically download models, but this may take 2-3 minutes on first deployment.

### Issue 3: Memory Limitations

**Cause**: Large AI models require significant memory
**Solution**: Streamlit Cloud provides sufficient resources, but initial loading may be slow.

### Issue 4: Import Path Issues

**Cause**: Different path resolution between local and cloud environments
**Solution**: Multiple import strategies implemented in `streamlit_app.py`

## 📊 Expected Deployment Timeline

1. **Repository Connection**: ~30 seconds
2. **Dependency Installation**: ~2-3 minutes
3. **Model Download**: ~3-5 minutes (first time only)
4. **App Initialization**: ~30 seconds
5. **Total First Deployment**: ~6-9 minutes

Subsequent deployments (after code changes) are much faster (~1-2 minutes).

## 🔍 Monitoring Deployment

### Streamlit Cloud Logs
Monitor the deployment logs for:
- ✅ Dependency installation progress
- ✅ Model download status
- ✅ Import success messages
- ❌ Any error messages

### Health Check
Once deployed, verify:
1. **App loads without errors**
2. **All UI components display**
3. **Text analysis works**
4. **Crisis resources are accessible**

## 🛠️ Advanced Configuration

### Environment Variables (Optional)
You can set these in Streamlit Cloud settings:
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`

### Custom Domain (Optional)
Streamlit Cloud provides a default URL, but you can configure a custom domain in the settings.

## 🆘 Getting Help

### If Deployment Fails:
1. **Check the logs** in Streamlit Cloud dashboard
2. **Verify all files** are in the repository
3. **Test locally** with `streamlit run streamlit_app.py`
4. **Open an issue** on GitHub with error details

### Support Resources:
- 📖 [Streamlit Documentation](https://docs.streamlit.io/streamlit-cloud)
- 🐛 [Report Issues](https://github.com/midlaj-muhammed/Mental-Health-Detector-AI-Application-Built/issues)
- 💬 [Community Support](https://discuss.streamlit.io/)

## ✅ Post-Deployment Checklist

After successful deployment:
- [ ] Test text analysis functionality
- [ ] Verify all visualizations work
- [ ] Check crisis resources links
- [ ] Test export functionality
- [ ] Confirm privacy disclaimers display
- [ ] Share the app URL responsibly

## 🔒 Security Considerations

- ✅ No user data is stored permanently
- ✅ All processing happens in memory
- ✅ Privacy disclaimers are prominent
- ✅ Crisis resources are easily accessible
- ✅ Ethical AI guidelines are followed

## 📱 Sharing Your App

Once deployed, you can share your app URL:
- **Public URL**: `https://your-app-name.streamlit.app`
- **Share responsibly** with appropriate context
- **Include disclaimers** about the tool's limitations
- **Provide crisis resources** alongside the app

---

**⚠️ Important**: This tool is designed to support, not replace, professional mental health care. Always include appropriate disclaimers when sharing.
