// nodejs_stt_test.js - Node.js'den AI Gateway STT API Test Script
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');

// API Base URL
const API_BASE_URL = 'http://localhost:8000/api/stt';

/**
 * 1. File Upload ile STT Test
 */
async function testFileUpload() {
    console.log('🎤 Testing File Upload STT...');
    
    try {
        const form = new FormData();
        form.append('file', fs.createReadStream('./test_audio.wav'));
        form.append('language', 'tr');
        form.append('timestamps', 'true');
        
        const response = await axios.post(
            `${API_BASE_URL}/transcribe/openai`,
            form,
            {
                headers: {
                    ...form.getHeaders(),
                    'Accept': 'application/json'
                },
                timeout: 30000
            }
        );
        
        console.log('✅ File Upload Success:');
        console.log('   Text:', response.data.text);
        console.log('   Provider:', response.data.provider);
        console.log('   Processing Time:', response.data.processing_time.toFixed(2) + 's');
        
        if (response.data.words) {
            console.log('   Word Timestamps:');
            response.data.words.forEach(word => {
                console.log(`     ${word.word}: ${word.start.toFixed(2)}s-${word.end.toFixed(2)}s`);
            });
        }
        
        return response.data;
        
    } catch (error) {
        console.error('❌ File Upload Error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * 2. Base64 ile STT Test
 */
async function testBase64Upload() {
    console.log('\\n🔐 Testing Base64 STT...');
    
    try {
        // Ses dosyasını base64'e çevir
        const audioBuffer = fs.readFileSync('./test_audio.wav');
        const audioBase64 = audioBuffer.toString('base64');
        
        const response = await axios.post(
            `${API_BASE_URL}/transcribe-base64/openai`,
            {
                audio_base64: audioBase64,
                format: 'wav',
                language: 'tr',
                timestamps: true,
                word_timestamps: true
            },
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                timeout: 30000
            }
        );
        
        console.log('✅ Base64 Upload Success:');
        console.log('   Text:', response.data.text);
        console.log('   Language:', response.data.language);
        console.log('   Duration:', response.data.duration?.toFixed(2) + 's');
        console.log('   Processing Time:', response.data.processing_time.toFixed(2) + 's');
        
        if (response.data.words) {
            console.log('   Word Timestamps:');
            response.data.words.forEach(word => {
                console.log(`     ${word.word}: ${word.start.toFixed(2)}s-${word.end.toFixed(2)}s`);
            });
        }
        
        return response.data;
        
    } catch (error) {
        console.error('❌ Base64 Upload Error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * 3. URL ile STT Test
 */
async function testUrlUpload() {
    console.log('\\n🌐 Testing URL STT...');
    
    try {
        // Test için bir örnek ses dosyası URL'i
        const testAudioUrl = 'https://www.soundjay.com/misc/sounds/bell-ringing-05.wav';
        
        const response = await axios.post(
            `${API_BASE_URL}/transcribe-url/openai`,
            {
                audio_url: testAudioUrl,
                language: 'en',  // English for this test
                timestamps: true,
                word_timestamps: true
            },
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                timeout: 45000  // URL download'ı için daha uzun timeout
            }
        );
        
        console.log('✅ URL Upload Success:');
        console.log('   Text:', response.data.text);
        console.log('   Language:', response.data.language);
        console.log('   Duration:', response.data.duration?.toFixed(2) + 's');
        console.log('   Processing Time:', response.data.processing_time.toFixed(2) + 's');
        
        return response.data;
        
    } catch (error) {
        console.error('❌ URL Upload Error:', error.response?.data || error.message);
        // URL test başarısız olursa devam et (test dosyası olmayabilir)
    }
}

/**
 * 4. Provider'ları ve API bilgilerini test et
 */
async function testProviders() {
    console.log('\\n🔧 Testing Providers...');
    
    try {
        // API bilgisi
        const apiInfo = await axios.get(`${API_BASE_URL}/`);
        console.log('✅ API Info:');
        console.log('   Version:', apiInfo.data.version);
        console.log('   Supported Providers:', apiInfo.data.supported_providers.join(', '));
        console.log('   Supported Formats:', apiInfo.data.supported_formats.join(', '));
        console.log('   Input Methods:', apiInfo.data.input_methods.join(', '));
        
        // Mevcut provider'lar
        const providers = await axios.get(`${API_BASE_URL}/providers`);
        console.log('\\n✅ Available Providers:');
        console.log('   Active Providers:', providers.data.providers.join(', '));
        console.log('   Default Provider:', providers.data.default);
        
        // Provider test
        if (providers.data.providers.includes('openai')) {
            const test = await axios.get(`${API_BASE_URL}/test/openai`);
            console.log('\\n✅ OpenAI Provider Test:');
            console.log('   Available:', test.data.available);
            console.log('   Supported Formats:', test.data.supported_formats.join(', '));
        }
        
    } catch (error) {
        console.error('❌ Provider Test Error:', error.response?.data || error.message);
    }
}

/**
 * Express.js Middleware Örneği
 */
function createSTTMiddleware() {
    return `
// express-stt-middleware.js
const multer = require('multer');
const axios = require('axios');

// Multer config (memory storage)
const upload = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 25 * 1024 * 1024 } // 25MB limit
});

// STT Middleware
const sttMiddleware = (provider = 'openai') => {
    return async (req, res, next) => {
        try {
            if (!req.file) {
                return res.status(400).json({ error: 'No audio file provided' });
            }
            
            // FormData oluştur
            const FormData = require('form-data');
            const form = new FormData();
            
            form.append('file', req.file.buffer, {
                filename: req.file.originalname,
                contentType: req.file.mimetype
            });
            
            form.append('language', req.body.language || 'tr');
            form.append('timestamps', req.body.timestamps || 'false');
            
            // STT API'ye gönder
            const response = await axios.post(
                \`http://localhost:8000/api/stt/transcribe/\${provider}\`,
                form,
                {
                    headers: form.getHeaders(),
                    timeout: 30000
                }
            );
            
            // Transcription'ı req object'e ekle
            req.transcription = response.data;
            next();
            
        } catch (error) {
            console.error('STT Middleware Error:', error);
            res.status(500).json({ 
                error: 'Transcription failed',
                details: error.response?.data || error.message 
            });
        }
    };
};

// Express route'da kullanım
app.post('/upload-audio', 
    upload.single('audio'), 
    sttMiddleware('openai'), 
    (req, res) => {
        res.json({
            success: true,
            transcription: req.transcription.text,
            words: req.transcription.words,
            processing_time: req.transcription.processing_time
        });
    }
);

module.exports = { sttMiddleware, upload };
`;
}

/**
 * Ana test fonksiyonu
 */
async function main() {
    console.log('🚀 AI Gateway STT API Test Suite');
    console.log('==================================\\n');
    
    try {
        // Ses dosyasının varlığını kontrol et
        if (!fs.existsSync('./test_audio.wav')) {
            console.log('❌ Test audio file not found: ./test_audio.wav');
            console.log('Please create a test audio file first.');
            return;
        }
        
        // 1. Provider testleri
        await testProviders();
        
        // 2. File upload test
        await testFileUpload();
        
        // 3. Base64 upload test
        await testBase64Upload();
        
        // 4. URL upload test (optional)
        await testUrlUpload();
        
        console.log('\\n✅ All tests completed successfully!');
        
        // 5. Express middleware örneğini göster
        console.log('\\n📋 Express.js Middleware Example:');
        console.log(createSTTMiddleware());
        
    } catch (error) {
        console.error('\\n❌ Test suite failed:', error.message);
        process.exit(1);
    }
}

// Script çalıştır
if (require.main === module) {
    main();
}

module.exports = {
    testFileUpload,
    testBase64Upload,
    testUrlUpload,
    testProviders
};