import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Iterator
from RAgents.llms.base import BaseLLM
from RAgents.llms.deepseek import DeepSeekLLM
from RAgents.llms.factory import LLMFactory


class TestBaseLLM:
    """测试 BaseLLM 抽象基类"""
    
    def test_base_llm_is_abstract(self):
        """测试 BaseLLM 不能直接实例化"""
        with pytest.raises(TypeError):
            BaseLLM("test_key", "test_model")
    
    def test_concrete_implementation(self):
        """测试具体的 BaseLLM 实现"""
        class TestLLM(BaseLLM):
            def generate(self, prompt: str, **kwargs) -> str:
                return f"Response: {prompt}"
            
            def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
                yield f"Stream: {prompt}"
        
        llm = TestLLM("test_key", "test_model", temperature=0.5)
        assert llm.api_key == "test_key"
        assert llm.model == "test_model"
        assert llm.config == {"temperature": 0.5}
        assert str(llm) == "TestLLM(model=test_model)"
        
        # 测试具体方法
        assert llm.generate("test") == "Response: test"
        assert list(llm.stream_generate("test")) == ["Stream: test"]
    
    def test_config_merge(self):
        """测试配置合并功能"""
        class TestLLM(BaseLLM):
            def generate(self, prompt: str, **kwargs) -> str:
                return ""
            
            def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
                return iter([])
        
        llm = TestLLM("key", "model", temperature=0.7, max_tokens=100)
        assert llm.config == {"temperature": 0.7, "max_tokens": 100}


class TestDeepSeekLLM:
    """测试 DeepSeekLLM 类"""
    
    def test_init_with_defaults(self):
        """测试使用默认参数初始化"""
        with patch('RAgents.llms.deepseek.OpenAI') as mock_openai:
            llm = DeepSeekLLM("test_api_key")
            
            assert llm.api_key == "test_api_key"
            assert llm.model == "deepseek-chat"
            assert llm.config == {}
            
            mock_openai.assert_called_once_with(
                api_key="test_api_key",
                base_url="https://api.deepseek.com"
            )
    
    def test_init_with_custom_params(self):
        """测试使用自定义参数初始化"""
        with patch('RAgents.llms.deepseek.OpenAI') as mock_openai:
            llm = DeepSeekLLM(
                api_key="custom_key",
                model="custom-model",
                base_url="https://custom.url",
                temperature=0.8,
                max_tokens=200
            )
            
            assert llm.api_key == "custom_key"
            assert llm.model == "custom-model"
            assert llm.config == {"temperature": 0.8, "max_tokens": 200}
            
            mock_openai.assert_called_once_with(
                api_key="custom_key",
                base_url="https://custom.url"
            )
    
    def test_generate_success(self):
        """测试成功生成响应"""
        with patch('RAgents.llms.deepseek.OpenAI') as mock_openai_class:
            # 设置 mock 客户端和响应
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            mock_client.chat.completions.create.return_value = mock_response
            
            llm = DeepSeekLLM("test_key", "test-model", temperature=0.5)
            result = llm.generate("Test prompt")
            
            assert result == "Test response"
            mock_client.chat.completions.create.assert_called_once_with(
                model="test-model",
                messages=[{"role": "user", "content": "Test prompt"}],
                temperature=0.5,
                timeout=60
            )
    
    def test_generate_with_overrides(self):
        """测试使用参数覆盖生成响应"""
        with patch('RAgents.llms.deepseek.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Response"
            mock_client.chat.completions.create.return_value = mock_response
            
            llm = DeepSeekLLM("test_key", "test-model", temperature=0.5)
            llm.generate("Test", max_tokens=100, temperature=0.8)
            
            # 验证调用参数包含覆盖值
            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args["temperature"] == 0.8  # 被覆盖
            assert call_args["max_tokens"] == 100  # 新增参数
    
    def test_stream_generate(self):
        """测试流式生成"""
        with patch('RAgents.llms.deepseek.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # 模拟流式响应
            def mock_stream():
                chunks = [
                    Mock(choices=[Mock(delta=Mock(content="Hello"))]),
                    Mock(choices=[Mock(delta=Mock(content=" world"))]),
                    Mock(choices=[Mock(delta=Mock(content=None))]),  # 空内容
                    Mock(choices=[Mock(delta=Mock(content="!" ))])
                ]
                for chunk in chunks:
                    yield chunk
            
            mock_client.chat.completions.create.return_value = mock_stream()
            
            llm = DeepSeekLLM("test_key")
            result = list(llm.stream_generate("Test"))
            
            assert result == ["Hello", " world", "!"]
            
            mock_client.chat.completions.create.assert_called_once_with(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Test"}],
                stream=True
            )
    
    def test_stream_generate_with_config(self):
        """测试使用配置的流式生成"""
        with patch('RAgents.llms.deepseek.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            mock_client.chat.completions.create.return_value = iter([])
            
            llm = DeepSeekLLM("test_key", temperature=0.7)
            list(llm.stream_generate("Test"))
            
            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args["temperature"] == 0.7


class TestLLMFactory:
    """测试 LLMFactory 工厂类"""
    
    def setup_method(self):
        """每个测试前重置工厂状态"""
        LLMFactory._providers.clear()
    
    def test_register_provider(self):
        """测试注册提供商"""
        class TestLLM:
            def __init__(self, api_key: str, **kwargs):
                self.api_key = api_key
        
        LLMFactory.register_provider("test", TestLLM)
        assert "test" in LLMFactory._providers
        assert LLMFactory._providers["test"] == TestLLM
    
    def test_register_provider_case_insensitive(self):
        """测试注册提供商时不区分大小写"""
        class TestLLM:
            pass
        
        LLMFactory.register_provider("TestProvider", TestLLM)
        assert "testprovider" in LLMFactory._providers
    
    def test_create_llm_success(self):
        """测试成功创建 LLM 实例"""
        class TestLLM:
            def __init__(self, api_key: str, model: str = None, **kwargs):
                self.api_key = api_key
                self.model = model or "default"
                self.kwargs = kwargs
        
        LLMFactory.register_provider("test", TestLLM)
        
        # 测试带模型
        llm1 = LLMFactory.create_llm("test", "key1", "model1", temp=0.5)
        assert llm1.api_key == "key1"
        assert llm1.model == "model1"
        assert llm1.kwargs == {"temp": 0.5}
        
        # 测试不带模型
        llm2 = LLMFactory.create_llm("test", "key2", temp=0.8)
        assert llm2.api_key == "key2"
        assert llm2.model == "default"
    
    def test_create_llm_lazy_loading(self):
        """测试延迟加载功能"""
        # 直接调用 _lazy_load_provider
        LLMFactory._lazy_load_provider("deepseek")
        assert "deepseek" in LLMFactory._providers
    
    def test_create_llm_case_insensitive(self):
        """测试创建 LLM 时不区分大小写"""
        LLMFactory._lazy_load_provider("deepseek")
        
        with patch('RAgents.llms.deepseek.OpenAI'):
            llm = LLMFactory.create_llm("DEEPSEEK", "test_key")
            assert isinstance(llm, DeepSeekLLM)
    
    def test_create_llm_unsupported_provider(self):
        """测试不支持的提供商"""
        with pytest.raises(ValueError, match="Unsupported LLM provider: nonexistent"):
            LLMFactory.create_llm("nonexistent", "key")
    
    def test_create_llm_lazy_load_failure(self):
        """测试延迟加载失败的情况"""
        # 模拟导入失败
        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            
            with pytest.raises(ValueError, match="Unsupported LLM provider: unknown"):
                LLMFactory.create_llm("unknown", "key")
    
    def test_list_providers(self):
        """测试列出提供商"""
        class TestLLM1:
            pass
        class TestLLM2:
            pass
        
        LLMFactory.register_provider("test1", TestLLM1)
        LLMFactory.register_provider("test2", TestLLM2)
        
        providers = LLMFactory.list_providers()
        assert set(providers) == {"test1", "test2"}
        assert isinstance(providers, list)
    
    def test_empty_list_providers(self):
        """测试空提供商列表"""
        providers = LLMFactory.list_providers()
        assert providers == []