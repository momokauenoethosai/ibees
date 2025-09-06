# サービスアカウントによる認証不要設定

ユーザーが認証なしでアプリケーションを使用できるようにする設定手順です。

## 1. サービスアカウントの作成

### Google Cloud Consoleで設定
1. [Google Cloud Console](https://console.cloud.google.com/) にアクセス
2. プロジェクトを選択
3. 左メニューから「IAMと管理」→「サービスアカウント」を選択
4. 「+サービスアカウントを作成」をクリック
5. 以下を設定：
   - サービスアカウント名: face-analysis-app
   - サービスアカウントID: 自動生成されたものを使用
   - 説明: Face Analysis Application Service Account
6. 「作成」をクリック

### 必要な権限を付与
1. ロールを選択：
   - Vertex AI ユーザー
   - BigQuery データ閲覧者
   - BigQuery ジョブユーザー
2. 「続行」→「完了」

### キーの作成
1. 作成したサービスアカウントをクリック
2. 「キー」タブを選択
3. 「鍵を追加」→「新しい鍵を作成」
4. 「JSON」を選択して「作成」
5. ダウンロードされたJSONファイルを `service-account.json` にリネーム

## 2. アプリケーションへの設定

```bash
# credentialsディレクトリに配置
mkdir -p webapp/credentials
mv ~/Downloads/[ダウンロードしたファイル名].json webapp/credentials/service-account.json

# 権限を制限
chmod 600 webapp/credentials/service-account.json
```

## 3. .gitignoreに追加

```
webapp/credentials/
!webapp/credentials/.gitkeep
```

## 4. 動作確認

アプリケーションを再起動すると、自動的にサービスアカウントを使用して認証されます。
ユーザーは認証なしで画像分析機能を使用できます。

## セキュリティに関する注意

- `service-account.json` は絶対にGitにコミットしないでください
- 本番環境では環境変数や秘密管理サービスを使用することを推奨
- APIの使用量に応じて課金が発生する可能性があります
- 必要に応じてレート制限を実装してください

## コスト管理

サービスアカウントを使用すると、全てのAPIコストがあなたのアカウントに請求されます。
以下の対策を検討してください：

1. **割り当て制限の設定**
   - Google Cloud Consoleで各APIの割り当てを制限
   
2. **予算アラートの設定**
   - 予期しない請求を防ぐため

3. **使用制限の実装**
   - アプリケーション側で1日あたりの使用回数を制限
   - IPアドレスごとの制限など